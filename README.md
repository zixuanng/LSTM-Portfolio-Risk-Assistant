# LSTM Portfolio Risk Assistant

A tool that uses LSTM models on historical price data to assess short-term risk/regime of your stock/ETF portfolio and visualize upcoming drawdown or volatility risk.

## Overview

This project implements an LSTM-based classifier to predict **volatility regimes** (low/medium/high) for a portfolio of stocks and ETFs over a 5-day horizon. It provides:

- **Per-ticker regime predictions** with probability distributions
- **Portfolio-level risk aggregation** based on user-defined weights
- **Interactive dashboard** for visualization and analysis

## Features

- 📊 **Data Collection**: Downloads historical OHLCV data via yfinance
- 🔧 **Feature Engineering**: Computes returns, volatility, RSI, MACD, moving averages, and more
- 🤖 **LSTM Model**: 2-layer LSTM classifier with dropout regularization
- 📈 **Baseline Comparison**: Compares against majority class, rule-based, and Random Forest baselines
- 🎯 **Dashboard**: Streamlit-based UI for ticker and portfolio views

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This will:

1. Download historical data for all configured tickers
2. Build features and labels
3. Train the LSTM model
4. Evaluate against baselines
5. Save the trained model to `models/`

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

## Project Structure

```
LSTM Portfolio Risk Assistant/
├── config.py         # Configuration constants (tickers, hyperparameters)
├── data.py           # Data download and management
├── features.py       # Feature engineering, labels, sequences
├── model.py          # LSTM architecture, training, evaluation
├── dashboard.py      # Streamlit UI + inference
├── main.py           # Training pipeline entry point
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── data/             # Downloaded data (created automatically)
│   ├── raw/          # OHLCV CSV files
│   └── processed/    # Feature datasets
├── models/           # Trained models (created automatically)
└── notebooks/        # Jupyter notebooks for exploration
```

## Configuration

Edit [`config.py`](config.py) to customize:

### Tickers

```python
TICKERS = {
    'us_etfs': ['SPY', 'QQQ', 'VTI'],
    'my_stocks': ['1155.KL', '5347.KL', '5225.KL', '1023.KL']
}
```

### Model Hyperparameters

```python
SEQUENCE_LENGTH = 30      # Days to look back
PREDICTION_HORIZON = 5    # Days ahead to predict
LSTM_UNITS = [64, 32]     # LSTM layer sizes
DROPOUT_RATE = 0.2        # Dropout probability
EPOCHS = 100              # Max training epochs
BATCH_SIZE = 32           # Training batch size
```

## Methodology

### Target Variable

The model predicts **5-day forward volatility regime**:

1. Compute realized volatility over the next 5 trading days
2. Bucket into tertiles: low (bottom 33%), medium (middle 33%), high (top 33%)
3. Train LSTM to classify the regime

### Features

For each day, the model uses the past 30 days of:

- **Returns**: Log returns, cumulative returns over multiple windows
- **Volatility**: Rolling standard deviation (5, 10, 20 days)
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Moving Averages**: SMA (10, 20, 50 days), price relative to MA
- **Volume**: Volume change, volume ratio to MA
- **Intraday Range**: High-Low spread

### Model Architecture

```
Input: (batch, 30, n_features)
    ↓
LSTM 64 units → Dropout 0.2
    ↓
LSTM 32 units → Dropout 0.2
    ↓
Dense 16 units (ReLU) → Dropout 0.2
    ↓
Output: 3 units (Softmax)
```

**Framework**: PyTorch (for better Python 3.14 compatibility)

### Training

- **Time-based split**: 70% train, 15% validation, 15% test (no random shuffle)
- **Early stopping**: Patience of 10 epochs on validation loss
- **Class weights**: Computed automatically for imbalanced classes
- **Learning rate reduction**: On validation loss plateau

## Dashboard Usage

### Ticker View

1. Select a ticker from the dropdown
2. View current regime prediction with probabilities
3. See price chart with volume

### Portfolio View

1. Adjust portfolio weights using sliders
2. View aggregate portfolio risk score
3. See risk contribution by ticker
4. Read summary recommendation

## Baseline Models

The LSTM is compared against:

1. **Majority Class**: Always predicts the most common regime
2. **Rule-Based**: Uses recent volatility thresholds
3. **Random Forest**: Flattened sequences with RF classifier

## Limitations

- **Educational purposes only**: Not financial advice
- **Daily data only**: No intraday signals
- **Historical patterns**: May not reflect future market conditions
- **Limited tickers**: Only configured stocks/ETFs

## Success Criteria

The model is considered successful if:

- LSTM accuracy > 55% on test set
- LSTM beats all baseline models
- Dashboard provides interpretable risk signals

## Troubleshooting

### "Model not found" error

Run `python main.py` first to train and save the model.

### Data download issues

- Check internet connection
- Some Malaysian stocks may have limited data on yfinance
- Try reducing `YEARS_OF_DATA` in config.py

### Training takes too long

- Reduce `EPOCHS` in config.py
- Reduce `LSTM_UNITS` to smaller values
- Use fewer tickers

## Future Improvements

- [ ] Add VIX/market sentiment features
- [ ] Implement drawdown prediction (alternative target)
- [ ] Add backtest visualization
- [ ] Support for custom ticker input
- [ ] Model ensemble with multiple horizons

## References

- [LSTM for Time Series Applications](https://towardsdatascience.com/five-practical-applications-of-the-lstm-model-for-time-series-with-code-a7aac0aa85c0/)
- [Stock Market Prediction with LSTM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9283061/)

## License

MIT License - for educational purposes only.

---

**Disclaimer**: This tool is for educational and research purposes only. It is not intended as financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
