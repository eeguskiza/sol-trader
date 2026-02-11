"""Tests for the data storage layer."""

from pathlib import Path

import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    """Point the singleton settings at a temporary directory for all tests."""
    monkeypatch.setattr(settings, "db_path", tmp_path)
    monkeypatch.setattr(settings, "data_path", tmp_path / "data")
    yield tmp_path


def test_duckdb_init_schema():
    from src.data.storage.duckdb_client import DuckDBClient

    client = DuckDBClient()
    client.init_schema()

    conn = client.connect()
    tables = conn.execute("SHOW TABLES").fetchdf()
    table_names = set(tables["name"].tolist())

    assert "ohlcv" in table_names
    assert "orderbook_snapshots" in table_names
    assert "funding_rates" in table_names

    client.close()


def test_duckdb_insert_and_query():
    from src.data.storage.duckdb_client import DuckDBClient

    client = DuckDBClient()
    client.init_schema()

    data = [
        {"timestamp": 1000, "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 500.0},
        {"timestamp": 2000, "open": 103.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 600.0},
    ]
    client.insert_ohlcv("SOL/USDT", "15m", data)

    df = client.get_ohlcv("SOL/USDT", "15m", 0, 3000)
    assert len(df) == 2
    assert df.iloc[0]["close"] == 103.0

    assert client.get_ohlcv_count() == 2

    client.close()


def test_sqlite_init_schema():
    from src.data.storage.sqlite_client import SQLiteClient

    client = SQLiteClient()
    client.init_schema()

    conn = client.connect()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}

    assert "trades" in table_names
    assert "state" in table_names
    assert "scraper_progress" in table_names

    client.close()


def test_sqlite_trade_lifecycle():
    from src.data.storage.sqlite_client import SQLiteClient

    client = SQLiteClient()
    client.init_schema()

    trade_id = client.save_trade({
        "timestamp": 1000,
        "symbol": "SOL/USDT",
        "side": "long",
        "size": 1.5,
        "entry_price": 100.0,
    })

    open_trades = client.get_open_trades("SOL/USDT")
    assert len(open_trades) == 1

    client.close_trade(trade_id, exit_price=110.0, pnl=15.0, pnl_pct=10.0)

    open_trades = client.get_open_trades("SOL/USDT")
    assert len(open_trades) == 0

    all_trades = client.get_all_trades()
    assert len(all_trades) == 1
    assert all_trades[0]["status"] == "closed"

    client.close()


def test_sqlite_scraper_progress():
    from src.data.storage.sqlite_client import SQLiteClient

    client = SQLiteClient()
    client.init_schema()

    assert client.get_scraper_progress("ohlcv", "SOL/USDT", "15m") is None

    client.set_scraper_progress("ohlcv", "SOL/USDT", "15m", 5000)
    assert client.get_scraper_progress("ohlcv", "SOL/USDT", "15m") == 5000

    client.set_scraper_progress("ohlcv", "SOL/USDT", "15m", 9000)
    assert client.get_scraper_progress("ohlcv", "SOL/USDT", "15m") == 9000

    client.close()
