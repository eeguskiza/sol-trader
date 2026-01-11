"""
Análisis del Dataset de Entrenamiento.

Script para inspeccionar el dataset etiquetado y visualizar
la calidad de los datos y la distribución de clases.
"""

import polars as pl
from pathlib import Path


def analyze_dataset():
    """Analizar el dataset de entrenamiento generado."""

    dataset_path = Path("data/processed/training_dataset.parquet")

    if not dataset_path.exists():
        print("❌ Dataset no encontrado. Ejecuta primero: python run_pipeline.py")
        return

    print("\n" + "="*70)
    print("📊 ANÁLISIS DEL DATASET DE ENTRENAMIENTO")
    print("="*70 + "\n")

    # Cargar dataset
    df = pl.read_parquet(dataset_path)

    # Información básica
    print("📁 Información básica:")
    print(f"   Total de muestras: {len(df):,}")
    print(f"   Total de features: {len(df.columns)}")
    print(f"   Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Rango temporal: {(df['timestamp'].max() - df['timestamp'].min()).days} días")

    # Distribución de clases
    print("\n🎯 Distribución de clases:")
    target_dist = df.group_by("target").count().sort("target")

    for row in target_dist.iter_rows(named=True):
        target = row['target']
        count = row['count']
        percentage = (count / len(df)) * 100
        label = "Buy Signal" if target == 1 else "No Trade"
        print(f"   {label} ({target}): {count:,} ({percentage:.2f}%)")

    # Estadísticas de precios
    print("\n💰 Estadísticas de precios:")
    price_stats = df.select([
        pl.col("close").min().alias("min"),
        pl.col("close").mean().alias("mean"),
        pl.col("close").max().alias("max"),
        pl.col("close").std().alias("std"),
    ])

    print(f"   Precio mínimo: ${price_stats['min'][0]:.2f}")
    print(f"   Precio medio: ${price_stats['mean'][0]:.2f}")
    print(f"   Precio máximo: ${price_stats['max'][0]:.2f}")
    print(f"   Desv. estándar: ${price_stats['std'][0]:.2f}")

    # Estadísticas de ATR (volatilidad)
    print("\n📈 Estadísticas de volatilidad (ATR):")
    atr_stats = df.select([
        pl.col("atr_14").min().alias("min"),
        pl.col("atr_14").mean().alias("mean"),
        pl.col("atr_14").max().alias("max"),
    ])

    print(f"   ATR mínimo: ${atr_stats['min'][0]:.4f}")
    print(f"   ATR medio: ${atr_stats['mean'][0]:.4f}")
    print(f"   ATR máximo: ${atr_stats['max'][0]:.4f}")

    # Análisis de umbrales
    print("\n🎯 Análisis de umbrales dinámicos:")
    threshold_stats = df.select([
        pl.col("threshold").min().alias("min"),
        pl.col("threshold").mean().alias("mean"),
        pl.col("threshold").max().alias("max"),
    ])

    print(f"   Umbral mínimo: ${threshold_stats['min'][0]:.4f}")
    print(f"   Umbral medio: ${threshold_stats['mean'][0]:.4f}")
    print(f"   Umbral máximo: ${threshold_stats['max'][0]:.4f}")

    # Análisis de retornos
    print("\n💵 Análisis de retornos futuros:")
    buy_signals = df.filter(pl.col("target") == 1)
    no_trade = df.filter(pl.col("target") == 0)

    if len(buy_signals) > 0:
        print(f"   Retorno medio (Buy Signals): {buy_signals['future_return_pct'].mean():.3f}%")
        print(f"   Retorno mediano (Buy): {buy_signals['future_return_pct'].median():.3f}%")
        print(f"   Retorno máximo (Buy): {buy_signals['future_return_pct'].max():.3f}%")

    if len(no_trade) > 0:
        print(f"   Retorno medio (No Trade): {no_trade['future_return_pct'].mean():.3f}%")
        print(f"   Retorno mediano (No Trade): {no_trade['future_return_pct'].median():.3f}%")

    # Columnas disponibles
    print("\n📦 Columnas disponibles (features):")
    feature_cols = [col for col in df.columns if col not in
                    ['timestamp', 'target', 'future_close', 'future_return',
                     'future_return_pct', 'threshold']]

    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")

    # Recomendaciones
    print("\n💡 Recomendaciones:")
    buy_pct = (len(buy_signals) / len(df)) * 100

    if buy_pct < 10:
        print("   ⚠️  Pocos Buy Signals (<10%)")
        print("      → Considera reducir atr_multiplier a 1.2 o 1.3")
        print("      → Esto generará más señales de compra")
    elif buy_pct > 40:
        print("   ⚠️  Muchos Buy Signals (>40%)")
        print("      → Considera aumentar atr_multiplier a 1.8 o 2.0")
        print("      → Esto filtrará señales de menor calidad")
    else:
        print("   ✅ Distribución balanceada (10-40%)")
        print("      → El dataset está listo para entrenamiento")

    # Vista previa de datos
    print("\n🔍 Vista previa (últimas 10 filas):")
    preview = df.select([
        "timestamp", "close", "rsi_14", "atr_14",
        "threshold", "future_return_pct", "target"
    ]).tail(10)

    print(preview)

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    analyze_dataset()
