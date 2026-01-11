# Solana Hybrid Trading Bot (Quant + LLM)

## Description

A high-frequency trading bot for Solana leveraging a hybrid architecture that combines quantitative and qualitative analysis for superior market predictions and execution.

### Hybrid Architecture

- **Quantitative Core:** Transformer-based time-series forecasting for price and volume predictions. Utilizes state-of-the-art deep learning models to identify patterns and trends in market data.

- **Qualitative Core:** Local LLM (Llama 3 via vLLM/Ollama) for real-time sentiment analysis. Processes social media, news, and on-chain activity to gauge market sentiment without relying on external APIs.

## Architecture

The system follows a strict separation of concerns:

```
Data Ingestion → Feature Engineering → Inference → Strategy → Execution
```

### Components

1. **Data Layer** (`src/scrapers/`)
   - Market data collection (price, volume, liquidity)
   - Sentiment data scraping (Twitter, Discord, news feeds)
   - On-chain metrics aggregation

2. **Quantitative Engine** (`src/quant_engine/`)
   - Transformer models for time-series forecasting
   - Feature extraction and normalization
   - Model training and evaluation pipelines

3. **Sentiment Engine** (`src/sentiment_engine/`)
   - Local LLM integration (vLLM/Ollama)
   - RAG (Retrieval-Augmented Generation) for context-aware analysis
   - Multi-source sentiment aggregation

4. **Strategy Layer** (`src/strategy/`)
   - Signal fusion (combining quant + qual signals)
   - Risk management and position sizing
   - Entry/exit logic and trade triggers

5. **Execution Layer** (`src/execution/`)
   - Solana RPC interaction
   - Transaction signing and submission
   - Order management and monitoring

6. **Utilities** (`src/utils/`)
   - Configuration management
   - Logging and monitoring
   - Helper functions

## Requirements

### System Requirements
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA support (RTX 50-series recommended for optimal performance)
- **RAM:** 32GB+ recommended for local LLM inference
- **Storage:** SSD with 100GB+ free space for data and model checkpoints

### Optional
- **Rust:** For optimized execution layer (optional but recommended for latency-critical operations)

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw Parquet/CSV market data
│   ├── processed/        # Normalized tensors and features
│   └── checkpoints/      # Model weights (.pt, .safetensors)
├── src/
│   ├── scrapers/         # Market and sentiment data collection
│   ├── quant_engine/     # Transformer/time-series models
│   ├── sentiment_engine/ # LLM/RAG integration
│   ├── strategy/         # Signal fusion and trading logic
│   ├── execution/        # Solana RPC and transaction handling
│   └── utils/            # Configuration, logging, helpers
├── notebooks/            # EDA and prototyping
├── tests/               # Unit and integration tests
├── config/              # Configuration files
└── requirements.txt     # Python dependencies
```

## Getting Started

*Documentation for setup and usage will be added as the project develops.*

## License

*To be determined*

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies carries significant risk. Always conduct your own research and never invest more than you can afford to lose.
