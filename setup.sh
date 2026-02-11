#!/bin/bash
set -e

echo "Setting up sol-trader environment..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p db data/checkpoints logs

# Copy env file if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please configure before running."
fi

# Initialize databases
python -c "from src.data.storage.duckdb_client import DuckDBClient; DuckDBClient().init_schema()"
python -c "from src.data.storage.sqlite_client import SQLiteClient; SQLiteClient().init_schema()"

echo "Setup complete. Run: source .venv/bin/activate && python cli.py --help"
