"""Application settings using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    mode: Literal["test", "live"] = "test"
    symbol: str = "SOL/USDT"
    timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h"]

    log_level: str = "INFO"
    db_path: Path = Path("./db")
    data_path: Path = Path("./data")

    initial_capital: float = 10000.0
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.02
    take_profit_ratio: float = 3.0
    signal_threshold: float = 0.3

    backtest_start: str = "2024-01-01"
    backtest_end: str = "2025-01-01"
    backtest_speed: float = 0

    binance_api_key: str = ""
    binance_secret: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
