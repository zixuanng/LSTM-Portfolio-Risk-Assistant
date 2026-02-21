"""
Feature engineering, label construction, and sequence building module.
Computes returns, volatility, technical indicators, and creates training sequences.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    RETURN_WINDOWS,
    VOLATILITY_WINDOWS,
    MA_WINDOWS,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    PREDICTION_HORIZON,
    REGIME_CLASSES,
    N_CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SCALER_PATH
)

from data import load_data, ensure_data_dir


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns.
    
    Args:
        prices: Series of prices
    
    Returns:
        Series of log returns
    """
    # Avoid division by zero and log of zero/negative
    ratio = prices / prices.shift(1)
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    log_ret = np.log(ratio)
    return log_ret.replace([np.inf, -np.inf], np.nan)


def compute_rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling standard deviation of returns.
    
    Args:
        returns: Series of returns
        window: Rolling window size
    
    Returns:
        Series of rolling volatility
    """
    return returns.rolling(window=window).std()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Series of prices
        window: RSI window (default 14)
    
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases: if avg_loss is 0, RSI should be 100
    rsi = rsi.fillna(100)
    
    return rsi


def compute_moving_averages(prices: pd.Series, windows: list) -> pd.DataFrame:
    """
    Compute simple moving averages for multiple windows.
    
    Args:
        prices: Series of prices
        windows: List of window sizes
    
    Returns:
        DataFrame with MA columns
    """
    mas = {}
    for window in windows:
        mas[f'MA_{window}'] = prices.rolling(window=window).mean()
    return pd.DataFrame(mas)


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Hist': histogram
    })


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    
    Args:
        prices: Series of prices
        window: Moving average window
        num_std: Number of standard deviations for bands
    
    Returns:
        DataFrame with Upper, Middle, Lower bands and Bandwidth
    """
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    # Avoid division by zero
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower,
        'BB_Bandwidth': bandwidth
    })


def compute_momentum(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Compute price momentum (rate of change).
    
    Args:
        prices: Series of prices
        window: Lookback window
    
    Returns:
        Series of momentum values
    """
    # Avoid division by zero
    ratio = prices / prices.shift(window)
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    return ratio - 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for a ticker's OHLCV data.
    
    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns
    
    Returns:
        DataFrame with all features
    """
    features = pd.DataFrame(index=df.index)
    
    # Use Close price for most calculations
    close = df['Close']
    
    # Log returns for multiple windows
    log_ret = compute_log_returns(close)
    features['Log_Return'] = log_ret
    
    for window in RETURN_WINDOWS[1:]:  # Skip 1 as it's the base
        features[f'Log_Return_{window}d'] = log_ret.rolling(window=window).sum()
    
    # Rolling volatility
    for window in VOLATILITY_WINDOWS:
        features[f'Volatility_{window}d'] = compute_rolling_volatility(log_ret, window)
    
    # Moving averages
    mas = compute_moving_averages(close, MA_WINDOWS)
    for col in mas.columns:
        features[col] = mas[col]
    
    # Price relative to MAs (avoid division by zero)
    for window in MA_WINDOWS:
        ma_col = features[f'MA_{window}'].replace(0, np.nan)
        features[f'Price_vs_MA{window}'] = close / ma_col - 1
    
    # RSI
    features['RSI'] = compute_rsi(close, RSI_WINDOW)
    
    # MACD
    macd = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    for col in macd.columns:
        features[col] = macd[col]
    
    # Bollinger Bands
    bb = compute_bollinger_bands(close)
    for col in bb.columns:
        features[col] = bb[col]
    
    # Momentum
    features['Momentum_10d'] = compute_momentum(close, 10)
    features['Momentum_20d'] = compute_momentum(close, 20)
    
    # Volume features (avoid division by zero)
    features['Volume_Change'] = df['Volume'].pct_change()
    features['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    vol_ma = features['Volume_MA_10'].replace(0, np.nan)
    features['Volume_Ratio'] = df['Volume'] / vol_ma
    
    # High-Low spread (intraday volatility proxy) - avoid division by zero
    close_safe = close.replace(0, np.nan)
    features['HL_Spread'] = (df['High'] - df['Low']) / close_safe
    
    # Drop Volume_MA_10 as we only need the ratio
    features = features.drop(columns=['Volume_MA_10'])
    
    # Replace any remaining inf values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


# =============================================================================
# LABEL CONSTRUCTION FUNCTIONS
# =============================================================================

def compute_future_volatility(returns: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Compute realized volatility over the next N days.
    
    Args:
        returns: Series of daily returns
        horizon: Number of days to look ahead
    
    Returns:
        Series of future volatility values
    """
    # Rolling std of next 'horizon' days
    # Use shift(-horizon+1) to get window starting from t+1
    future_vol = returns.shift(-horizon+1).rolling(window=horizon).std()
    return future_vol


def create_regime_labels(future_vol: pd.Series, method: str = 'tertiles') -> pd.Series:
    """
    Create regime labels based on future volatility.
    
    Args:
        future_vol: Series of future volatility values
        method: 'tertiles' for 3 equal-sized groups
    
    Returns:
        Series of regime labels (0=low, 1=medium, 2=high)
    """
    # Remove NaN values for percentile calculation
    valid_vol = future_vol.dropna()
    
    if method == 'tertiles':
        # Compute tertile boundaries
        low_threshold = valid_vol.quantile(1/3)
        high_threshold = valid_vol.quantile(2/3)
        
        labels = pd.Series(index=future_vol.index, dtype=float)
        labels[future_vol <= low_threshold] = 0  # low
        labels[(future_vol > low_threshold) & (future_vol <= high_threshold)] = 1  # medium
        labels[future_vol > high_threshold] = 2  # high
        
        return labels.astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_dataset(ticker: str) -> pd.DataFrame:
    """
    Load data for a ticker, build features, and add regime labels.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        DataFrame with features and regime_label column
    """
    # Load raw data
    df = load_data(ticker)
    
    if df is None:
        print(f"No data available for {ticker}")
        return None
    
    # Build features
    features = build_features(df)
    
    # Compute future volatility for labels
    returns = features['Log_Return']
    future_vol = compute_future_volatility(returns, PREDICTION_HORIZON)
    
    # Create regime labels
    features['regime_label'] = create_regime_labels(future_vol)
    
    # Add ticker column for reference
    features['ticker'] = ticker
    
    return features


# =============================================================================
# SEQUENCE BUILDING FUNCTIONS
# =============================================================================

def build_sequences(features: np.ndarray, labels: np.ndarray, seq_length: int = 30) -> tuple:
    """
    Build sliding window sequences for LSTM.
    
    Args:
        features: 2D array of features (n_samples, n_features)
        labels: 1D array of labels (n_samples,)
        seq_length: Length of each sequence
    
    Returns:
        X: 3D array (n_sequences, seq_length, n_features)
        y: 1D array (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(labels[i+seq_length])
    
    return np.array(X), np.array(y)


def time_based_split(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float) -> tuple:
    """
    Split data by time (no random shuffling).
    
    Args:
        X: Feature sequences
        y: Labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_all_data(tickers: list = None, save_scaler: bool = True) -> tuple:
    """
    Prepare sequences for all tickers combined.
    
    Args:
        tickers: List of ticker symbols
        save_scaler: Whether to save the fitted scaler
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    from config import ALL_TICKERS
    
    if tickers is None:
        tickers = ALL_TICKERS
    
    all_features = []
    all_labels = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = prepare_dataset(ticker)
        
        if df is None:
            continue
        
        # Drop NaN rows (from rolling calculations and future labels)
        df_clean = df.dropna()
        
        if len(df_clean) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
            print(f"  Skipping {ticker}: insufficient data ({len(df_clean)} rows)")
            continue
        
        # Separate features and labels
        feature_cols = [col for col in df_clean.columns if col not in ['regime_label', 'ticker']]
        features = df_clean[feature_cols].values
        labels = df_clean['regime_label'].values
        
        all_features.append(features)
        all_labels.append(labels)
        
        print(f"  Added {len(features)} samples from {ticker}")
    
    # Combine all tickers
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"\nTotal samples: {len(all_features)}")
    
    # Handle infinity and NaN values
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scaler on all data (before sequence building)
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Save scaler for later use
    if save_scaler:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved to {SCALER_PATH}")
    
    # Build sequences
    X, y = build_sequences(all_features_scaled, all_labels, SEQUENCE_LENGTH)
    
    print(f"Total sequences: {len(X)}")
    
    # Time-based split
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        X, y, TRAIN_RATIO, VAL_RATIO
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Feature names for reference
    feature_names = [col for col in df_clean.columns if col not in ['regime_label', 'ticker']]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def get_latest_sequence(ticker: str, scaler: StandardScaler, seq_length: int = None) -> np.ndarray:
    """
    Build sequence from most recent data for inference.
    
    Args:
        ticker: Ticker symbol
        scaler: Fitted StandardScaler
        seq_length: Sequence length (default from config)
    
    Returns:
        3D array (1, seq_length, n_features) for model input
    """
    if seq_length is None:
        seq_length = SEQUENCE_LENGTH
    
    # Prepare dataset (without labels for latest data)
    df = load_data(ticker)
    
    if df is None:
        return None
    
    # Build features
    features = build_features(df)
    
    # Get the most recent seq_length rows
    features_recent = features.iloc[-seq_length:]
    
    # Check for NaN
    if features_recent.isna().any().any():
        # Fill NaN with forward fill, then backward fill
        features_recent = features_recent.ffill().bfill()
    
    # Scale
    features_scaled = scaler.transform(features_recent.values)
    
    # Reshape for LSTM (1, seq_length, n_features)
    sequence = features_scaled.reshape(1, seq_length, -1)
    
    return sequence


if __name__ == "__main__":
    # Test feature engineering
    print("=" * 50)
    print("Testing feature engineering...")
    print("=" * 50)
    
    # Prepare data for all tickers
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_all_data()
    
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    # Check label distribution
    print(f"\nLabel distribution (train):")
    for i, label in enumerate(REGIME_CLASSES):
        count = (y_train == i).sum()
        print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
