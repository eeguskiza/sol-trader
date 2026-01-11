"""
Complete Data Pipeline Example.

This script demonstrates the full workflow from raw market data
to labeled training dataset ready for Transformer model training.

Pipeline stages:
1. Fetch historical market data (OHLCV)
2. Calculate technical indicators
3. Build labeled dataset with ATR-based targets
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.market_scraper import MarketScraper
from src.processors.technical_indicators import FeatureEngineer
from src.quant_engine.dataset_builder import DatasetBuilder


def run_complete_pipeline(
    days_back: int = 180,
    symbol: str = "SOL/USDT",
    timeframe: str = "15m"
):
    """
    Run the complete data pipeline.

    Args:
        days_back: Number of days of historical data to fetch
        symbol: Trading pair symbol
        timeframe: Candle timeframe
    """
    print("="*70)
    print("SOLANA TRADING BOT - DATA PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Symbol:     {symbol}")
    print(f"  Timeframe:  {timeframe}")
    print(f"  History:    {days_back} days")
    print("\n" + "="*70 + "\n")

    # Stage 1: Fetch market data
    print("STAGE 1/3: Fetching Market Data")
    print("-" * 70)

    scraper = MarketScraper(
        symbol=symbol,
        timeframe=timeframe,
        output_dir="data/raw/market"
    )

    start_date = datetime.now() - timedelta(days=days_back)

    try:
        # Fetch historical data
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to now...")
        market_data = scraper.fetch_historical(start_date)

        print(f"✅ Fetched {len(market_data)} candles")
        print(f"   Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return False

    # Stage 2: Calculate technical indicators
    print("\n" + "="*70)
    print("STAGE 2/3: Calculating Technical Indicators")
    print("-" * 70)

    engineer = FeatureEngineer()

    try:
        print("Adding technical indicators...")
        print("  - RSI (14)")
        print("  - ATR (14)")
        print("  - EMA (50, 200)")
        print("  - Bollinger Bands (20, 2σ)")

        features_df = engineer.add_all_features(market_data)

        print(f"✅ Added {len(features_df.columns) - len(market_data.columns)} indicators")
        print(f"   Total features: {len(features_df.columns)}")

        # Save processed data
        processed_path = Path("data/processed")
        processed_path.mkdir(parents=True, exist_ok=True)

        output_filename = f"{symbol.replace('/', '_')}_{timeframe}_processed.parquet"
        output_path = processed_path / output_filename

        features_df.write_parquet(output_path, compression="snappy")
        print(f"   Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return False

    # Stage 3: Build labeled dataset
    print("\n" + "="*70)
    print("STAGE 3/3: Building Labeled Dataset")
    print("-" * 70)

    builder = DatasetBuilder(
        processed_dir="data/processed",
        lookahead_candles=4,  # 1 hour on 15m timeframe
        atr_multiplier=1.5    # Volatility-adjusted threshold
    )

    try:
        print("Creating target variable with ATR-based dynamic thresholds...")
        print(f"  Lookahead: 4 candles (1 hour)")
        print(f"  Strategy: Buy if future_price > current_price + (1.5 × ATR)")

        training_dataset = builder.build_training_dataset(
            input_filename=output_filename,
            output_filename="training_dataset.parquet"
        )

        print("✅ Training dataset created successfully!")

    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        return False

    # Pipeline complete
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\n📁 Output files:")
    print(f"   Raw data:        data/raw/market/")
    print(f"   Processed data:  data/processed/{output_filename}")
    print(f"   Training data:   data/processed/training_dataset.parquet")
    print("\n🚀 Next steps:")
    print("   1. Analyze class balance and adjust ATR multiplier if needed")
    print("   2. Split dataset into train/validation/test sets")
    print("   3. Begin Transformer model training")
    print("\n" + "="*70 + "\n")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build training dataset for SOL trading bot")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data to fetch")
    parser.add_argument("--symbol", type=str, default="SOL/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")

    args = parser.parse_args()

    success = run_complete_pipeline(
        days_back=args.days,
        symbol=args.symbol,
        timeframe=args.timeframe
    )

    sys.exit(0 if success else 1)
