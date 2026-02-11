# Solana Trading Bot

Quantitative trading system for SOL/USDT using transformer-based price prediction and order flow analysis.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (optional, for training)
- 16GB+ RAM recommended

## Quick Start

```bash
# Setup environment
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

# Download historical data
python cli.py download --start 2024-01-01

# Run backtest
python cli.py backtest --start 2024-06-01 --end 2024-12-01

# Check status
python cli.py status
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download historical OHLCV data |
| `train` | Train prediction model |
| `backtest` | Run backtest simulation |
| `paper` | Paper trading (live data, simulated execution) |
| `live` | Live trading (use with caution) |
| `status` | Show system status |

## Configuration

Edit `.env` file:

```
MODE=test
SYMBOL=SOL/USDT
INITIAL_CAPITAL=10000.0
```

## Architecture

```
Data Layer (DuckDB) -> Features -> Model -> Strategy -> Execution
```

- **Data**: OHLCV multi-timeframe, order book depth, funding rates
- **Features**: Technical indicators, order flow imbalance
- **Model**: Transformer encoder for price action
- **Strategy**: Signal fusion with risk management
- **Execution**: Backtest simulator / Live adapter

## Project Structure

```
sol-trader/
├── cli.py                  # Entry point
├── config/settings.py      # Pydantic settings
├── src/
│   ├── data/
│   │   ├── scrapers/       # OHLCV, orderbook, funding
│   │   ├── storage/        # DuckDB + SQLite
│   │   └── replay/         # Async data replay
│   ├── features/           # Technical + orderflow
│   ├── models/             # Transformer + inference
│   ├── strategy/           # Signals, risk, executor
│   ├── adapters/           # Backtest / paper / live
│   └── backtest/           # Engine, simulator, metrics
├── db/                     # DuckDB + SQLite files
├── data/checkpoints/       # Model weights
└── tests/
```

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies carries significant risk.
