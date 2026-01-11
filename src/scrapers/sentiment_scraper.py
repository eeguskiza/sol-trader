"""
Sentiment Data Scraper using CryptoPanic API.

This module provides production-ready functionality for fetching cryptocurrency
news and sentiment data with robust error handling and Parquet storage.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional, List, Dict, Any

import polars as pl
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentScraper:
    """
    High-performance sentiment data scraper for cryptocurrency news.

    Uses CryptoPanic API for real-time news aggregation and stores
    structured data in Parquet format for efficient retrieval.
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        currencies: str = "SOL",
        output_dir: str = "data/raw/sentiment"
    ):
        """
        Initialize the sentiment scraper.

        Args:
            api_key: CryptoPanic API key (reads from CRYPTOPANIC_API_KEY env var if None)
            currencies: Comma-separated list of currency codes (default: SOL)
            output_dir: Directory for Parquet file storage

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CryptoPanic API key required. "
                "Set CRYPTOPANIC_API_KEY environment variable or pass api_key parameter."
            )

        self.currencies = currencies
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SentimentScraper initialized for currencies: {currencies}")

    def fetch_recent_news(
        self,
        limit: int = 100,
        filter_type: str = "hot",
        kind: str = "news"
    ) -> pl.DataFrame:
        """
        Fetch recent news articles for configured currencies.

        Args:
            limit: Maximum number of articles to fetch (API processes internally)
            filter_type: Filter type - 'rising', 'hot', 'bullish', 'bearish', 'important', 'saved', 'lol'
            kind: Content kind - 'news' or 'media' or 'all'

        Returns:
            Polars DataFrame with columns: timestamp, title, source, url, sentiment_label, currencies

        Raises:
            requests.RequestException: If API request fails
        """
        logger.info(f"Fetching {filter_type} {kind} for {self.currencies}")

        articles = []
        next_page = None
        fetched_count = 0

        try:
            while fetched_count < limit:
                # Fetch batch
                data = self._fetch_page(
                    filter_type=filter_type,
                    kind=kind,
                    next_page=next_page
                )

                if not data or "results" not in data:
                    logger.warning("No results returned from API")
                    break

                results = data["results"]
                if not results:
                    logger.info("No more articles available")
                    break

                articles.extend(results)
                fetched_count += len(results)

                logger.info(f"Fetched {len(results)} articles. Total: {fetched_count}")

                # Check for next page
                next_page = data.get("next")
                if not next_page or fetched_count >= limit:
                    break

                # Rate limiting (free tier: 50 requests/day, be conservative)
                sleep(1)

            # Convert to DataFrame
            df = self._articles_to_dataframe(articles[:limit])

            # Save to Parquet
            self._save_data(df)

            logger.info(f"Successfully fetched {len(df)} articles")
            return df

        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            raise

    def fetch_historical_batch(
        self,
        days_back: int = 7,
        filter_type: str = "all"
    ) -> pl.DataFrame:
        """
        Fetch historical news data for backtesting.

        Args:
            days_back: Number of days to fetch (limited by API tier)
            filter_type: Filter type for news selection

        Returns:
            Polars DataFrame with historical sentiment data
        """
        logger.info(f"Fetching historical data for last {days_back} days")

        # Note: Historical data access depends on API tier
        # Free tier has limited history; paid tiers have full access
        return self.fetch_recent_news(limit=1000, filter_type=filter_type)

    def _fetch_page(
        self,
        filter_type: str = "hot",
        kind: str = "news",
        next_page: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Fetch a single page of news articles with retry logic.

        Args:
            filter_type: Filter type for news
            kind: Content kind
            next_page: URL for next page (pagination)
            max_retries: Maximum number of retry attempts

        Returns:
            JSON response as dictionary

        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                # Use next_page URL if available (already includes auth token)
                if next_page:
                    response = requests.get(next_page, timeout=10)
                else:
                    params = {
                        "auth_token": self.api_key,
                        "currencies": self.currencies,
                        "filter": filter_type,
                        "kind": kind,
                        "public": "true"
                    }
                    response = requests.get(self.BASE_URL, params=params, timeout=10)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout. Retrying in {wait_time}s...")
                sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = 60  # Wait 1 minute
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    sleep(wait_time)
                elif response.status_code == 401:
                    raise ValueError("Invalid API key")
                else:
                    logger.error(f"HTTP error: {e}")
                    raise

            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request error: {e}. Retrying in {wait_time}s...")
                sleep(wait_time)

        raise requests.RequestException(f"Failed to fetch data after {max_retries} attempts")

    def _articles_to_dataframe(self, articles: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        Convert raw articles to structured Polars DataFrame.

        Args:
            articles: List of article dictionaries from API

        Returns:
            Polars DataFrame with clean schema
        """
        if not articles:
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime,
                "title": pl.Utf8,
                "source": pl.Utf8,
                "url": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "currencies": pl.Utf8,
                "votes_positive": pl.Int64,
                "votes_negative": pl.Int64,
                "votes_important": pl.Int64,
            })

        # Extract relevant fields
        data = {
            "timestamp": [
                datetime.fromisoformat(a["published_at"].replace("Z", "+00:00"))
                for a in articles
            ],
            "title": [a.get("title", "") for a in articles],
            "source": [
                a.get("source", {}).get("title", "Unknown") if isinstance(a.get("source"), dict)
                else "Unknown"
                for a in articles
            ],
            "url": [a.get("url", "") for a in articles],
            "sentiment_label": [
                self._extract_sentiment(a.get("votes", {}))
                for a in articles
            ],
            "currencies": [
                ",".join([c["code"] for c in a.get("currencies", [])])
                for a in articles
            ],
            "votes_positive": [a.get("votes", {}).get("positive", 0) for a in articles],
            "votes_negative": [a.get("votes", {}).get("negative", 0) for a in articles],
            "votes_important": [a.get("votes", {}).get("important", 0) for a in articles],
        }

        df = pl.DataFrame(data)

        # Add sentiment score placeholder (to be filled by LLM later)
        df = df.with_columns(
            pl.lit(0.0).alias("sentiment_score")  # Placeholder: -1 to 1 scale
        )

        return df

    def _extract_sentiment(self, votes: Dict[str, int]) -> str:
        """
        Extract sentiment label from vote counts.

        Args:
            votes: Dictionary with 'positive', 'negative', 'important' counts

        Returns:
            Sentiment label: 'bullish', 'bearish', or 'neutral'
        """
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)

        if positive > negative:
            return "bullish"
        elif negative > positive:
            return "bearish"
        else:
            return "neutral"

    def _save_data(self, df: pl.DataFrame) -> None:
        """
        Save sentiment data to Parquet file.

        Args:
            df: Polars DataFrame to save
        """
        if df.is_empty():
            logger.warning("No data to save")
            return

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_{self.currencies}_{timestamp}.parquet"
        filepath = self.output_dir / filename

        df.write_parquet(filepath, compression="snappy")
        logger.info(f"Saved {len(df)} articles to {filepath}")

    def load_all_data(self) -> pl.DataFrame:
        """
        Load all sentiment Parquet files from output directory.

        Returns:
            Combined Polars DataFrame sorted by timestamp
        """
        parquet_files = list(self.output_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("No Parquet files found")
            return pl.DataFrame()

        logger.info(f"Loading {len(parquet_files)} sentiment files")

        # Load all files and concatenate
        dfs = [pl.read_parquet(f) for f in parquet_files]
        combined_df = pl.concat(dfs).unique(subset=["url"]).sort("timestamp")

        logger.info(f"Loaded {len(combined_df)} total articles")
        return combined_df


if __name__ == "__main__":
    # Example usage
    try:
        scraper = SentimentScraper()

        # Fetch recent hot news
        news_df = scraper.fetch_recent_news(limit=50, filter_type="hot")

        print(f"Fetched {len(news_df)} news articles")
        print("\nSample data:")
        print(news_df.select(["timestamp", "title", "sentiment_label", "source"]).head(10))

        # Show sentiment distribution
        print("\nSentiment distribution:")
        print(news_df.group_by("sentiment_label").count())

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use this scraper, get a free API key from:")
        print("https://cryptopanic.com/developers/api/")
        print("Then set it in your .env file: CRYPTOPANIC_API_KEY=your_key_here")
