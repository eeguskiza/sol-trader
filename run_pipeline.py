"""
Pipeline Completo de Datos para SOL Trading Bot.

Este script conecta todos los pasos:
1. Lee datos RAW de market_scraper
2. Calcula indicadores técnicos
3. Genera dataset etiquetado con targets

Ejecutar: python run_pipeline.py
"""

import os
import sys
import glob
from pathlib import Path

import polars as pl

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.technical_indicators import FeatureEngineer
from src.quant_engine.dataset_builder import DatasetBuilder


def get_latest_raw_file():
    """Buscar el archivo parquet más reciente en data/raw/market/"""
    files = glob.glob("data/raw/market/*.parquet")

    if not files:
        raise FileNotFoundError(
            "❌ No hay datos raw!\n"
            "   Ejecuta primero: python src/scrapers/market_scraper.py"
        )

    # Devolver el más reciente por fecha de modificación
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def main():
    print("\n" + "="*70)
    print("🚀 PIPELINE DE DATOS - SOL TRADING BOT")
    print("="*70 + "\n")

    # PASO 1: Identificar archivo raw más reciente
    print("PASO 1/3: Buscando datos raw...")
    print("-" * 70)

    try:
        raw_file = get_latest_raw_file()
        print(f"✅ Archivo detectado: {raw_file}")

        # Cargar datos
        df_raw = pl.read_parquet(raw_file)
        print(f"   Registros: {len(df_raw):,}")
        print(f"   Columnas: {df_raw.columns}")
        print(f"   Período: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error leyendo archivo raw: {e}")
        sys.exit(1)

    # PASO 2: Calcular Indicadores Técnicos
    print("\n" + "="*70)
    print("PASO 2/3: Calculando indicadores técnicos...")
    print("-" * 70)

    try:
        engineer = FeatureEngineer()

        print("Aplicando indicadores:")
        print("  • RSI (14)")
        print("  • ATR (14) - Crucial para el umbral dinámico")
        print("  • EMA (50, 200)")
        print("  • Bollinger Bands (20, 2σ)")

        # Aplicar todos los indicadores
        df_processed = engineer.add_all_features(df_raw)

        initial_cols = len(df_raw.columns)
        final_cols = len(df_processed.columns)
        added_features = final_cols - initial_cols

        print(f"\n✅ Indicadores calculados:")
        print(f"   Columnas originales: {initial_cols}")
        print(f"   Columnas finales: {final_cols}")
        print(f"   Features añadidas: {added_features}")

        # Guardar datos procesados
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        processed_filename = "SOL_USDT_15m_indicators.parquet"
        processed_path = processed_dir / processed_filename

        df_processed.write_parquet(processed_path, compression="snappy")

        file_size_mb = processed_path.stat().st_size / (1024 * 1024)
        print(f"   Guardado en: {processed_path}")
        print(f"   Tamaño: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ Error calculando indicadores: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # PASO 3: Generar Dataset Etiquetado
    print("\n" + "="*70)
    print("PASO 3/3: Generando dataset etiquetado...")
    print("-" * 70)

    try:
        print("Configuración de la estrategia 'Sniper':")
        print("  • Lookahead: 4 candles (1 hora en 15m)")
        print("  • Umbral: ATR × 1.5 (dinámico por volatilidad)")
        print("  • Target: 1 si precio_futuro > precio_actual + umbral")
        print()

        builder = DatasetBuilder(
            processed_dir="data/processed",
            lookahead_candles=4,   # 1 hora
            atr_multiplier=1.5     # Umbral balanceado
        )

        # Construir dataset de entrenamiento
        # Nota: build_training_dataset espera solo el nombre del archivo, no la ruta completa
        final_dataset = builder.build_training_dataset(
            input_filename=processed_filename,
            output_filename="training_dataset.parquet",
            validate=True
        )

        print(f"\n✅ Dataset final creado exitosamente!")
        print(f"   Ubicación: data/processed/training_dataset.parquet")
        print(f"   Registros finales: {len(final_dataset):,}")

    except Exception as e:
        print(f"❌ Error generando dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # RESUMEN FINAL
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\n📁 Archivos generados:")
    print(f"   1. {processed_path}")
    print(f"   2. data/processed/training_dataset.parquet")
    print("\n🎯 Próximos pasos:")
    print("   1. Revisar la distribución de clases (Buy Signals %)")
    print("   2. Ajustar atr_multiplier si es necesario (1.2-2.0)")
    print("   3. Dividir en train/val/test sets")
    print("   4. ¡Comenzar a entrenar el modelo Transformer!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
