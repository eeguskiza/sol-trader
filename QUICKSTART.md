# 🚀 Guía Rápida - Pipeline de Datos

## Resumen del Flujo de Trabajo

```
1. market_scraper.py  →  data/raw/market/*.parquet
                         ↓
2. run_pipeline.py    →  procesa raw + calcula indicadores
                         ↓
3. Resultado Final    →  data/processed/training_dataset.parquet
```

---

## 📋 Pasos para Generar el Dataset

### Paso 1: Instalar Dependencias

```bash
cd sol-trader
pip install -r requirements.txt
```

### Paso 2: Descargar Datos de Mercado (Solo primera vez)

Si no tienes datos raw, descárgalos:

```bash
python src/scrapers/market_scraper.py
```

Esto descargará ~6 meses de datos históricos de SOL/USDT en intervalos de 15 minutos.

### Paso 3: Ejecutar el Pipeline Completo

```bash
python run_pipeline.py
```

**¿Qué hace este script?**
1. ✅ Lee el archivo raw más reciente de `data/raw/market/`
2. ✅ Calcula indicadores técnicos (RSI, ATR, EMA, Bollinger Bands)
3. ✅ Guarda datos procesados en `data/processed/SOL_USDT_15m_indicators.parquet`
4. ✅ Genera targets con umbral ATR dinámico
5. ✅ Guarda dataset final en `data/processed/training_dataset.parquet`

### Paso 4: Analizar el Dataset Generado

```bash
python analyze_dataset.py
```

Esto mostrará:
- Distribución de clases (Buy Signals vs No Trade)
- Estadísticas de precios y volatilidad
- Retornos esperados por clase
- Recomendaciones para ajustar parámetros

---

## 📊 Resultados Esperados

### Archivos Generados

```
sol-trader/
├── data/
│   ├── raw/market/
│   │   └── SOL_USDT_15m_*.parquet         # Datos crudos de CCXT
│   └── processed/
│       ├── SOL_USDT_15m_indicators.parquet # Con indicadores técnicos
│       └── training_dataset.parquet        # Dataset etiquetado (FINAL)
```

### Distribución Típica de Clases

Con `atr_multiplier=1.5` (configuración por defecto):

| Clase | Porcentaje | Descripción |
|-------|-----------|-------------|
| **0** (No Trade) | ~85-90% | Precio no se moverá significativamente |
| **1** (Buy Signal) | ~10-15% | Precio subirá más que el umbral ATR |

---

## ⚙️ Ajustando Parámetros

### Si hay POCOS Buy Signals (<10%)

Edita `run_pipeline.py` línea 102:

```python
builder = DatasetBuilder(
    lookahead_candles=4,
    atr_multiplier=1.2  # Era 1.5, ahora más agresivo
)
```

### Si hay MUCHOS Buy Signals (>40%)

```python
builder = DatasetBuilder(
    lookahead_candles=4,
    atr_multiplier=2.0  # Era 1.5, ahora más conservador
)
```

Después de cambiar, vuelve a ejecutar:
```bash
python run_pipeline.py
```

---

## 🔧 Configuración del .env (Opcional)

Para sentiment scraping necesitas una API key de CryptoPanic:

```bash
# Copia el ejemplo
cp .env.example .env

# Edita y añade tu API key
nano .env
```

```env
CRYPTOPANIC_API_KEY=tu_api_key_aqui
```

---

## 📚 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| `run_pipeline.py` | **Script principal** - Ejecuta todo el flujo |
| `analyze_dataset.py` | Analiza el dataset generado |
| `src/scrapers/market_scraper.py` | Descarga datos de Binance |
| `src/processors/technical_indicators.py` | Calcula RSI, ATR, EMA, BB |
| `src/quant_engine/dataset_builder.py` | Genera targets con ATR |

---

## 🎯 Próximos Pasos

Una vez tengas `training_dataset.parquet`:

1. **Dividir en train/val/test:**
   ```python
   import polars as pl

   df = pl.read_parquet("data/processed/training_dataset.parquet")

   # División temporal (80/10/10)
   n = len(df)
   train = df[:int(n*0.8)]
   val = df[int(n*0.8):int(n*0.9)]
   test = df[int(n*0.9):]
   ```

2. **Entrenar modelo Transformer** (próxima fase)

3. **Backtesting y evaluación**

---

## 🐛 Troubleshooting

### Error: "No module named 'polars'"
```bash
pip install polars numpy ccxt requests python-dotenv
```

### Error: "No hay datos raw"
```bash
python src/scrapers/market_scraper.py
```

### Error: "Missing required columns"
Asegúrate de que el archivo raw tenga las columnas: `timestamp, open, high, low, close, volume`

---

## 📈 Interpretación de Resultados

### Ejemplo de Salida del Pipeline:

```
🎯 Class Distribution:
   Total Samples:       7,261
   Buy Signals (1):     746 (10.27%)    ← ¡Balanceado!
   No Trade (0):        6,515 (89.73%)

📈 Performance Metrics:
   Avg Return (Buy):    1.232%    ← Señales positivas
   Avg Return (No):     -0.160%   ← No trade = sin ganancia
```

**Interpretación:**
- ✅ **10.27% Buy Signals**: Balanceado, no hay overfitting
- ✅ **+1.232% Avg Return**: Las señales de compra son rentables en promedio
- ✅ **-0.160% No Trade**: Correctamente identifica momentos sin movimiento

---

## 🚀 Comando Todo-en-Uno

```bash
# Instalar, descargar datos y procesar (primera vez)
pip install -r requirements.txt && \
python src/scrapers/market_scraper.py && \
python run_pipeline.py && \
python analyze_dataset.py
```

---

## 📞 Soporte

Si encuentras problemas:
1. Verifica que todos los paquetes estén instalados
2. Asegúrate de tener datos raw descargados
3. Revisa los logs para errores específicos
4. Consulta `docs/LABELING_STRATEGY.md` para detalles técnicos
