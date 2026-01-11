"""
Market Data Scraper for SOL/USDT using CCXT.

This module provides production-ready functionality for fetching OHLCV data
from Binance with robust error handling, rate limiting, and Parquet storage.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Optional

import ccxt
import polars as pl
from ccxt.base.errors import RateLimitExceeded, NetworkError, ExchangeError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketScraper:
    """
    High-performance market data scraper for SOL/USDT on Binance.

    Uses CCXT for reliable data access and Polars for efficient storage.
    Handles rate limiting, pagination, and incremental saves automatically.
    """

    def __init__(
        self,
        symbol: str = "SOL/USDT",
        timeframe: str = "15m",
        output_dir: str = "data/raw/market"
    ):
        """
        Initialize the market scraper.

        Args:
            symbol: Trading pair symbol (default: SOL/USDT)
            timeframe: Candle timeframe (default: 15m)
            output_dir: Directory for Parquet file storage
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Built-in rate limiting
            'options': {
                'defaultType': 'spot',  # Spot trading
            }
        })

        logger.info(f"MarketScraper initialized for {symbol} on {timeframe} timeframe")

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV data with pagination.

        Args:
            start_date: Start date for historical data
            end_date: End date (default: now)
            batch_size: Number of candles per API request (max 1000 for Binance)

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ExchangeError: If exchange API returns an error
            NetworkError: If network connection fails
        """
        if end_date is None:
            end_date = datetime.now()

        logger.info(f"Fetching historical data from {start_date} to {end_date}")

        all_candles = []
        current_timestamp = int(start_date.timestamp() * 1000)  # Convert to milliseconds
        end_timestamp = int(end_date.timestamp() * 1000)

        try:
            while current_timestamp < end_timestamp:
                # Fetch batch with retry logic
                candles = self._fetch_with_retry(
                    since=current_timestamp,
                    limit=batch_size
                )

                if not candles:
                    logger.warning("No more data available")
                    break

                all_candles.extend(candles)

                # Update timestamp for next batch (last candle timestamp + 1ms)
                current_timestamp = candles[-1][0] + 1

                logger.info(f"Fetched {len(candles)} candles. Total: {len(all_candles)}")

                # Save incrementally every 10,000 candles
                if len(all_candles) >= 10000:
                    self._save_batch(all_candles)
                    all_candles = []

                # Respect rate limits (built-in, but extra safety)
                sleep(self.exchange.rateLimit / 1000)

            # Save remaining candles
            if all_candles:
                self._save_batch(all_candles)

            # Load and return all saved data
            return self._load_all_data()

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def fetch_latest(self, limit: int = 100) -> pl.DataFrame:
        """
        Fetch the most recent candles for live inference.

        Args:
            limit: Number of recent candles to fetch (max 1000)

        Returns:
            Polars DataFrame with recent OHLCV data

        Raises:
            ExchangeError: If exchange API returns an error
        """
        logger.info(f"Fetching latest {limit} candles")

        try:
            candles = self._fetch_with_retry(limit=limit)
            df = self._candles_to_dataframe(candles)

            logger.info(f"Successfully fetched {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            raise

    def _fetch_with_retry(
        self,
        since: Optional[int] = None,
        limit: int = 1000,
        max_retries: int = 5
    ) -> list:
        """
        Fetch OHLCV data with exponential backoff retry logic.

        Args:
            since: Timestamp in milliseconds (optional)
            limit: Number of candles to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            List of OHLCV candles

        Raises:
            ExchangeError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=limit
                )
                return candles

            except RateLimitExceeded as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time}s...")
                sleep(wait_time)

            except NetworkError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Network error: {e}. Retrying in {wait_time}s...")
                sleep(wait_time)

            except ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise

        raise ExchangeError(f"Failed to fetch data after {max_retries} attempts")

    def _candles_to_dataframe(self, candles: list) -> pl.DataFrame:
        """
        Convert raw OHLCV candles to Polars DataFrame.

        Args:
            candles: List of [timestamp, open, high, low, close, volume]

        Returns:
            Polars DataFrame with proper schema
        """
        if not candles:
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64
            })

        df = pl.DataFrame(
            {
                "timestamp": [c[0] for c in candles],
                "open": [c[1] for c in candles],
                "high": [c[2] for c in candles],
                "low": [c[3] for c in candles],
                "close": [c[4] for c in candles],
                "volume": [c[5] for c in candles],
            }
        ).with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("timestamp")
        )

        return df

    def _save_batch(self, candles: list) -> None:
        """
        Save a batch of candles to Parquet file.

        Args:
            candles: List of OHLCV candles
        """
        df = self._candles_to_dataframe(candles)

        # Generate filename with timestamp range
        start_ts = df["timestamp"].min().strftime("%Y%m%d_%H%M%S")
        end_ts = df["timestamp"].max().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol.replace('/', '_')}_{self.timeframe}_{start_ts}_to_{end_ts}.parquet"

        filepath = self.output_dir / filename
        df.write_parquet(filepath, compression="snappy")

        logger.info(f"Saved {len(df)} candles to {filepath}")

    def _load_all_data(self) -> pl.DataFrame:
        """
        Load all Parquet files from output directory.

        Returns:
            Combined Polars DataFrame sorted by timestamp
        """
        parquet_files = list(self.output_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("No Parquet files found")
            return pl.DataFrame()

        logger.info(f"Loading {len(parquet_files)} Parquet files")

        # Load all files and concatenate
        dfs = [pl.read_parquet(f) for f in parquet_files]
        combined_df = pl.concat(dfs).unique().sort("timestamp")

        logger.info(f"Loaded {len(combined_df)} total candles")
        return combined_df


if __name__ == "__main__":
    # Example usage
    scraper = MarketScraper()

    # Fetch last 6 months of data
    start_date = datetime.now() - timedelta(days=180)
    historical_data = scraper.fetch_historical(start_date)

    print(f"Fetched {len(historical_data)} historical candles")
    print(historical_data.head())

    # Fetch latest 100 candles
    latest_data = scraper.fetch_latest(limit=100)
    print(f"\nLatest {len(latest_data)} candles:")
    print(latest_data.tail())
