"""CLI entry point for the Solana Trading Bot."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

app = typer.Typer(help="Solana Trading Bot - Quantitative Trading System")
console = Console()


def _create_progress() -> Progress:
    """Create a Rich Progress bar with spinner, bar, percentage and ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@app.command()
def download(
    symbol: str = typer.Option("SOL/USDT", help="Trading pair"),
    start: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, help="End date (YYYY-MM-DD), defaults to today"),
    timeframes: str = typer.Option("1m,5m,15m,1h,4h", help="Comma-separated timeframes"),
):
    """Download historical market data."""
    from src.data.scrapers.ohlcv import OHLCVScraper
    from src.utils.logging import setup_logging

    setup_logging()

    console.print(Panel(f"Downloading {symbol} data from {start}", title="Data Download"))

    scraper = OHLCVScraper()

    setup_logging(console_output=False)
    with _create_progress() as progress:
        for tf in timeframes.split(","):
            tf = tf.strip()
            task = progress.add_task(f"Downloading {tf}...", total=None)

            def on_progress(current: int, total: int, _task=task) -> None:
                progress.update(_task, completed=current, total=total)

            count = scraper.scrape(symbol, tf, start, end, on_progress=on_progress)
            progress.update(task, description=f"Downloaded {tf}: {count} candles")
    setup_logging()

    console.print("[green]Download complete[/green]")


@app.command()
def train(
    epochs: int = typer.Option(100, help="Maximum training epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    patience: int = typer.Option(15, help="Early stopping patience"),
    start: str = typer.Option("2024-01-01", help="Training data start date"),
    end: str = typer.Option("2025-01-01", help="Training data end date"),
):
    """Train the price prediction model."""
    from datetime import datetime

    from config.settings import settings as _settings
    from src.utils.logging import setup_logging
    from src.models.dataset import create_dataloaders
    from src.models.transformer import PriceActionTransformer
    from src.models.trainer import Trainer, get_device

    setup_logging()

    console.print(Panel(
        f"Training Configuration\n"
        f"Data: {start} to {end}\n"
        f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}",
        title="Model Training",
    ))

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)

    console.print("Loading data...")
    train_loader, val_loader, class_weights = create_dataloaders(
        symbol=_settings.symbol,
        timeframe="15m",
        start_ts=start_ts,
        end_ts=end_ts,
        batch_size=batch_size,
    )

    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[-1]

    console.print(f"Input dimension: {input_dim}")
    console.print(f"Device: {get_device()}")

    model = PriceActionTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    )

    console.print(f"Model parameters: {model.count_parameters():,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=lr,
    )

    setup_logging(console_output=False)
    with _create_progress() as progress:
        task = progress.add_task("Training...", total=epochs)

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        history = trainer.train(epochs=epochs, patience=patience, on_progress=on_progress)
    setup_logging()

    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
    best_f1 = max(history["val_f1"])

    table = Table(title="Training Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Best Epoch", str(best_epoch))
    table.add_row("Best Val Loss", f"{min(history['val_loss']):.4f}")
    table.add_row("Best Val F1", f"{best_f1:.3f}")
    table.add_row("Final Val Acc", f"{history['val_acc'][-1]:.3f}")
    table.add_row("Checkpoint", str(_settings.data_path / "checkpoints" / "best_model.pt"))

    console.print(table)


@app.command()
def backtest(
    start: str = typer.Option("2024-01-01", help="Backtest start date"),
    end: str = typer.Option("2025-01-01", help="Backtest end date"),
    capital: float = typer.Option(10000.0, help="Initial capital"),
    speed: float = typer.Option(0, help="Simulation speed (0=instant, >0=realtime multiplier)"),
):
    """Run backtest simulation."""
    import os

    os.environ["BACKTEST_START"] = start
    os.environ["BACKTEST_END"] = end
    os.environ["INITIAL_CAPITAL"] = str(capital)
    os.environ["BACKTEST_SPEED"] = str(speed)

    from config.settings import Settings

    _settings = Settings()

    from src.utils.logging import setup_logging
    from src.adapters.backtest import BacktestAdapter
    from src.strategy.executor import StrategyExecutor
    from src.backtest.metrics import calculate_metrics

    setup_logging()

    console.print(
        Panel(f"Backtest: {start} to {end}\nCapital: ${capital:,.2f}", title="Backtest")
    )

    setup_logging(console_output=False)
    with _create_progress() as progress:
        task = progress.add_task("Running backtest...", total=None)

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        async def _run():
            adapter = BacktestAdapter()
            strategy = StrategyExecutor(adapter)
            await strategy.run(on_progress=on_progress)
            return adapter.get_trade_history(), _settings.initial_capital

        trades, initial = asyncio.run(_run())
        metrics = calculate_metrics(trades, initial)
    setup_logging()

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Win Rate", f"{metrics.win_rate:.1f}%")
    table.add_row("Total PnL", f"${metrics.total_pnl:,.2f}")
    table.add_row("Total Return", f"{metrics.total_pnl_pct:.2f}%")
    table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

    console.print(table)


@app.command()
def paper():
    """Run paper trading with live data (simulated execution)."""
    console.print("[yellow]Paper trading not yet implemented[/yellow]")


@app.command()
def live():
    """Run live trading (real execution). Use with caution."""
    console.print("[red]Live trading not yet implemented[/red]")
    console.print("[yellow]This will execute real trades. Make sure you understand the risks.[/yellow]")


@app.command()
def status():
    """Show system status and database info."""
    from src.data.storage.duckdb_client import DuckDBClient
    from src.data.storage.sqlite_client import SQLiteClient

    console.print(Panel("System Status", title="Status"))

    try:
        duckdb = DuckDBClient()
        count = duckdb.get_ohlcv_count()
        console.print(f"[green]DuckDB: {count:,} OHLCV records[/green]")
    except Exception as e:
        console.print(f"[red]DuckDB: Error - {e}[/red]")

    try:
        sqlite = SQLiteClient()
        conn = sqlite.connect()
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        console.print(f"[green]SQLite: {trade_count:,} trades[/green]")
    except Exception as e:
        console.print(f"[red]SQLite: Error - {e}[/red]")


if __name__ == "__main__":
    app()
