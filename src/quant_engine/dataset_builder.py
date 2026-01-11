"""
Dataset Builder for Transformer-based Trading Model.

This module creates labeled datasets for binary classification using
a "Sniper" strategy with ATR-based dynamic thresholds for volatility adaptation.

The target predicts whether price will move significantly upward within 1 hour (4x15min candles).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Build labeled datasets for quantitative trading models.

    Uses ATR-based dynamic thresholds to adapt to market volatility conditions.
    In calm markets, smaller moves trigger buy signals. In volatile markets,
    we demand larger moves to ensure statistical significance.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        lookahead_candles: int = 4,
        atr_multiplier: float = 1.5,
        atr_column: str = "atr_14"
    ):
        """
        Initialize the dataset builder.

        Args:
            processed_dir: Directory containing processed parquet files with indicators
            lookahead_candles: Number of candles to look ahead for target (default: 4 = 1 hour)
            atr_multiplier: Multiplier for ATR threshold (default: 1.5)
            atr_column: Name of ATR column to use (default: 'atr_14')
        """
        self.processed_dir = Path(processed_dir)
        self.lookahead_candles = lookahead_candles
        self.atr_multiplier = atr_multiplier
        self.atr_column = atr_column

        logger.info(
            f"DatasetBuilder initialized - Lookahead: {lookahead_candles} candles, "
            f"ATR multiplier: {atr_multiplier}"
        )

    def build_training_dataset(
        self,
        input_filename: str,
        output_filename: str = "training_dataset.parquet",
        validate: bool = True
    ) -> pl.DataFrame:
        """
        Build a complete training dataset with labels.

        Args:
            input_filename: Name of processed parquet file (with technical indicators)
            output_filename: Name for output parquet file
            validate: Whether to validate data quality (default: True)

        Returns:
            Polars DataFrame with features and target column

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        input_path = self.processed_dir / input_filename
        output_path = self.processed_dir / output_filename

        logger.info(f"Loading data from {input_path}")

        # Load processed data
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pl.read_parquet(input_path)
        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows} rows")

        # Validate required columns
        self._validate_columns(df)

        # Create target variable
        df = self._create_target(df)

        # Clean data
        df = self._clean_data(df)

        # Validate data quality
        if validate:
            self._validate_dataset(df)

        # Generate statistics
        stats = self._generate_statistics(df, initial_rows)

        # Save dataset
        self._save_dataset(df, output_path)

        # Print summary
        self._print_summary(stats)

        return df

    def _validate_columns(self, df: pl.DataFrame) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"close", self.atr_column}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Ensure technical indicators have been calculated."
            )

        logger.info(f"Validation passed - Found {len(df.columns)} columns")

    def _create_target(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create binary target variable using ATR-based dynamic thresholds.

        Strategy Logic (Sniper Entry):
        - Look ahead N candles (e.g., 4 = 1 hour on 15m timeframe)
        - Calculate future return
        - Label = 1 if future_close > current_close + (ATR_multiplier * current_ATR)
        - Label = 0 otherwise

        This approach:
        1. Adapts to volatility (higher threshold in volatile markets)
        2. Filters noise (only significant moves trigger buys)
        3. Balances risk/reward (ATR represents typical price movement)

        Args:
            df: DataFrame with close prices and ATR

        Returns:
            DataFrame with added columns: future_close, future_return, threshold, target
        """
        logger.info(f"Creating target with {self.lookahead_candles}-candle lookahead")

        df = df.with_columns([
            # Shift close price backwards to get future values
            # shift(-4) means "get the value 4 rows ahead"
            pl.col("close").shift(-self.lookahead_candles).alias("future_close"),
        ])

        df = df.with_columns([
            # Calculate absolute future return (price difference)
            (pl.col("future_close") - pl.col("close")).alias("future_return"),

            # Calculate dynamic threshold based on current volatility
            # In calm markets (low ATR), we need smaller moves
            # In volatile markets (high ATR), we demand larger moves
            (pl.col(self.atr_column) * self.atr_multiplier).alias("threshold"),
        ])

        df = df.with_columns([
            # Binary classification target
            # 1 = Buy signal (price will move up significantly)
            # 0 = No trade (insufficient expected movement)
            pl.when(pl.col("future_return") > pl.col("threshold"))
              .then(1)
              .otherwise(0)
              .cast(pl.Int32)
              .alias("target"),
        ])

        # Add percentage return for analysis
        df = df.with_columns([
            ((pl.col("future_return") / pl.col("close")) * 100).alias("future_return_pct"),
        ])

        logger.info("Target variable created successfully")
        return df

    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean dataset by removing invalid values.

        Removes:
        1. NaN values (from indicator warm-up periods like EMA-200)
        2. Infinite values (from division by zero, etc.)
        3. Rows where future data is unavailable (last N rows)

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        logger.info("Cleaning data...")

        # Remove rows with any NaN values
        df_clean = df.drop_nulls()
        nan_removed = initial_rows - len(df_clean)

        # Check for infinite values in numeric columns
        numeric_cols = [col for col in df_clean.columns if df_clean[col].dtype in [pl.Float64, pl.Float32]]

        # Remove infinite values
        for col in numeric_cols:
            df_clean = df_clean.filter(pl.col(col).is_finite())

        infinite_removed = len(df_clean) - (initial_rows - nan_removed)

        # Remove rows where target is null (last N candles without future data)
        df_clean = df_clean.filter(pl.col("target").is_not_null())

        logger.info(
            f"Removed {nan_removed} rows with NaN, "
            f"{abs(infinite_removed)} rows with infinite values"
        )
        logger.info(f"Clean dataset: {len(df_clean)} rows")

        return df_clean

    def _validate_dataset(self, df: pl.DataFrame) -> None:
        """
        Validate dataset quality and warn about potential issues.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If critical issues are detected
        """
        logger.info("Validating dataset quality...")

        # Check if dataset is too small
        if len(df) < 1000:
            logger.warning(
                f"Dataset has only {len(df)} samples. "
                "Consider using more historical data for robust model training."
            )

        # Check for extreme class imbalance
        target_counts = df["target"].value_counts().sort("target")
        total = len(df)

        if len(target_counts) < 2:
            raise ValueError("Dataset has only one class. Cannot train binary classifier.")

        minority_class_pct = (target_counts["count"].min() / total) * 100

        if minority_class_pct < 5:
            logger.warning(
                f"Severe class imbalance detected: minority class is only {minority_class_pct:.2f}%. "
                "Consider adjusting ATR multiplier or using class weights during training."
            )
        elif minority_class_pct < 20:
            logger.info(
                f"Moderate class imbalance: minority class is {minority_class_pct:.2f}%. "
                "This is acceptable but consider using class weights."
            )

        # Check for data leakage (target should not be all 1s or all 0s in any window)
        # Check rolling windows for suspicious patterns
        window_size = 100
        if len(df) >= window_size:
            rolling_target = df.select([
                pl.col("target").rolling_mean(window_size=window_size).alias("rolling_target_mean")
            ])

            max_mean = rolling_target["rolling_target_mean"].max()
            min_mean = rolling_target["rolling_target_mean"].min()

            if max_mean == 1.0 or min_mean == 0.0:
                logger.warning("Detected suspicious patterns in target variable. Check for data leakage.")

        logger.info("Dataset validation complete")

    def _generate_statistics(self, df: pl.DataFrame, initial_rows: int) -> dict:
        """
        Generate comprehensive dataset statistics.

        Args:
            df: Final cleaned DataFrame
            initial_rows: Number of rows before cleaning

        Returns:
            Dictionary with statistics
        """
        target_counts = df["target"].value_counts().sort("target")
        total_samples = len(df)

        buy_signals = target_counts.filter(pl.col("target") == 1)["count"][0] if len(target_counts) > 0 else 0
        no_trade = target_counts.filter(pl.col("target") == 0)["count"][0] if len(target_counts) > 0 else 0

        buy_pct = (buy_signals / total_samples) * 100 if total_samples > 0 else 0
        no_trade_pct = (no_trade / total_samples) * 100 if total_samples > 0 else 0

        # Calculate average returns for each class
        avg_return_buy = df.filter(pl.col("target") == 1)["future_return_pct"].mean()
        avg_return_no_trade = df.filter(pl.col("target") == 0)["future_return_pct"].mean()

        # Calculate average threshold
        avg_threshold = df["threshold"].mean()
        avg_atr = df[self.atr_column].mean()

        stats = {
            "initial_rows": initial_rows,
            "final_rows": total_samples,
            "rows_removed": initial_rows - total_samples,
            "removal_pct": ((initial_rows - total_samples) / initial_rows * 100) if initial_rows > 0 else 0,
            "buy_signals": buy_signals,
            "no_trade": no_trade,
            "buy_pct": buy_pct,
            "no_trade_pct": no_trade_pct,
            "avg_return_buy": avg_return_buy,
            "avg_return_no_trade": avg_return_no_trade,
            "avg_threshold": avg_threshold,
            "avg_atr": avg_atr,
            "feature_count": len(df.columns) - 5,  # Exclude target and derived columns
        }

        return stats

    def _save_dataset(self, df: pl.DataFrame, output_path: Path) -> None:
        """
        Save dataset to Parquet file.

        Args:
            df: DataFrame to save
            output_path: Path for output file
        """
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        df.write_parquet(output_path, compression="snappy")
        logger.info(f"Dataset saved to {output_path}")

        # Log file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

    def _print_summary(self, stats: dict) -> None:
        """
        Print formatted summary statistics.

        Args:
            stats: Dictionary with statistics
        """
        print("\n" + "="*70)
        print("TRAINING DATASET SUMMARY")
        print("="*70)
        print(f"\n📊 Data Cleaning:")
        print(f"   Initial rows:        {stats['initial_rows']:,}")
        print(f"   Final rows:          {stats['final_rows']:,}")
        print(f"   Removed:             {stats['rows_removed']:,} ({stats['removal_pct']:.2f}%)")
        print(f"\n🎯 Class Distribution:")
        print(f"   Total Samples:       {stats['final_rows']:,}")
        print(f"   Buy Signals (1):     {stats['buy_signals']:,} ({stats['buy_pct']:.2f}%)")
        print(f"   No Trade (0):        {stats['no_trade']:,} ({stats['no_trade_pct']:.2f}%)")
        print(f"\n📈 Performance Metrics:")
        print(f"   Avg Return (Buy):    {stats['avg_return_buy']:.3f}%")
        print(f"   Avg Return (No):     {stats['avg_return_no_trade']:.3f}%")
        print(f"\n🔧 Strategy Parameters:")
        print(f"   Lookahead:           {self.lookahead_candles} candles (1 hour)")
        print(f"   ATR Multiplier:      {self.atr_multiplier}x")
        print(f"   Avg Threshold:       ${stats['avg_threshold']:.4f}")
        print(f"   Avg ATR:             ${stats['avg_atr']:.4f}")
        print(f"\n📦 Features:")
        print(f"   Feature count:       {stats['feature_count']}")
        print("="*70 + "\n")

        # Warnings
        if stats['buy_pct'] < 10:
            print("⚠️  WARNING: Very low buy signal rate. Consider:")
            print("   - Reducing ATR multiplier (currently {:.1f})".format(self.atr_multiplier))
            print("   - Using different technical indicators")
            print("   - Adjusting lookahead period\n")
        elif stats['buy_pct'] > 40:
            print("⚠️  WARNING: High buy signal rate may indicate overfitting. Consider:")
            print("   - Increasing ATR multiplier (currently {:.1f})".format(self.atr_multiplier))
            print("   - Adding more conservative filters\n")

    def analyze_thresholds(self, df: pl.DataFrame) -> None:
        """
        Analyze threshold distribution for debugging.

        Args:
            df: DataFrame with threshold column
        """
        print("\n" + "="*70)
        print("THRESHOLD ANALYSIS")
        print("="*70)

        threshold_stats = df.select([
            pl.col("threshold").min().alias("min"),
            pl.col("threshold").quantile(0.25).alias("q25"),
            pl.col("threshold").median().alias("median"),
            pl.col("threshold").quantile(0.75).alias("q75"),
            pl.col("threshold").max().alias("max"),
        ])

        print(threshold_stats)
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    import sys

    # For demonstration, we need to create sample data
    # In production, you would use actual processed market data
    print("Dataset Builder - Example Usage\n")

    # Check if processed file exists
    builder = DatasetBuilder(
        processed_dir="data/processed",
        lookahead_candles=4,
        atr_multiplier=1.5
    )

    # Try to find a processed file
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        parquet_files = list(processed_dir.glob("*.parquet"))

        if parquet_files and parquet_files[0].name != "training_dataset.parquet":
            input_file = parquet_files[0].name
            print(f"Found processed file: {input_file}")

            try:
                dataset = builder.build_training_dataset(
                    input_filename=input_file,
                    output_filename="training_dataset.parquet"
                )

                print(f"✅ Successfully created training dataset!")
                print(f"   Output: data/processed/training_dataset.parquet")

            except Exception as e:
                print(f"❌ Error building dataset: {e}")
                sys.exit(1)
        else:
            print("❌ No processed parquet files found.")
            print("\nTo create a training dataset:")
            print("1. Run market_scraper.py to fetch OHLCV data")
            print("2. Run technical_indicators.py to add features")
            print("3. Run this script to build labeled dataset")
    else:
        print("❌ Directory 'data/processed' not found.")
        print("\nPlease ensure you have:")
        print("1. Fetched market data using market_scraper.py")
        print("2. Generated technical indicators using technical_indicators.py")
