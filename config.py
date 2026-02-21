"""
Configuration constants for LSTM Portfolio Risk Assistant.
All hyperparameters and settings are defined here for easy modification.
"""

# =============================================================================
# TICKER CONFIGURATION
# =============================================================================

TICKERS = {
    'us_etfs': ['SPY', 'QQQ', 'VTI'],
    'my_stocks': ['1155.KL', '5347.KL', '5225.KL', '1023.KL']
}

# Flattened list of all tickers
ALL_TICKERS = TICKERS['us_etfs'] + TICKERS['my_stocks']

# Default portfolio weights (equal weight if not specified)
DEFAULT_WEIGHTS = {ticker: 1.0 / len(ALL_TICKERS) for ticker in ALL_TICKERS}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Directory paths
DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'

# Data parameters
YEARS_OF_DATA = 10  # How many years of historical data to download

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Sequence length for LSTM (number of trading days to look back)
SEQUENCE_LENGTH = 30

# Rolling windows for feature calculation
RETURN_WINDOWS = [1, 5, 10, 20]
VOLATILITY_WINDOWS = [5, 10, 20]
MA_WINDOWS = [10, 20, 50]

# Technical indicator parameters
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# =============================================================================
# LABEL CONSTRUCTION
# =============================================================================

# Prediction horizon (how many days ahead to predict volatility regime)
PREDICTION_HORIZON = 5

# Regime class labels
REGIME_CLASSES = ['low', 'medium', 'high']
N_CLASSES = len(REGIME_CLASSES)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LSTM architecture
LSTM_UNITS = [64, 32]  # Units in each LSTM layer
DROPOUT_RATE = 0.2
DENSE_UNITS = 16

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping patience

# =============================================================================
# TRAIN/VAL/TEST SPLIT
# =============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Should sum to 1.0 with TRAIN_RATIO and VAL_RATIO

# =============================================================================
# BASELINE THRESHOLDS
# =============================================================================

# For rule-based baseline: thresholds to classify volatility
# These will be computed from training data percentiles
VOLATILITY_LOW_PERCENTILE = 33
VOLATILITY_HIGH_PERCENTILE = 67

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

# Default date range for charts (in days)
DEFAULT_CHART_DAYS = 252  # ~1 year of trading days

# Color scheme for regimes
REGIME_COLORS = {
    'low': '#2ECC71',      # Green
    'medium': '#F39C12',   # Orange/Yellow
    'high': '#E74C3C'      # Red
}

# =============================================================================
# MODEL FILE PATHS
# =============================================================================

MODEL_PATH = 'models/lstm_regime.pt'
SCALER_PATH = 'models/scaler.pkl'
