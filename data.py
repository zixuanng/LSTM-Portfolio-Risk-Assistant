"""
Data collection and management module.
Downloads OHLCV data from yfinance and manages local storage.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from config import (
    DATA_DIR,
    YEARS_OF_DATA,
    ALL_TICKERS
)


def ensure_data_dir():
    """Ensure data directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def get_ticker_filepath(ticker: str) -> str:
    """Get the file path for a ticker's data."""
    # Replace problematic characters in filename (e.g., .KL -> _KL)
    safe_name = ticker.replace('.', '_')
    return os.path.join(DATA_DIR, f"{safe_name}.csv")


def download_ticker_data(ticker: str, years: int = None, start_date: str = None) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single ticker.
    
    Args:
        ticker: Stock/ETF ticker symbol
        years: Number of years of historical data (default from config)
        start_date: Optional specific start date (YYYY-MM-DD format)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if years is None:
        years = YEARS_OF_DATA
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} from {start_date}...")
    
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        
        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return None
        
        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Keep only standard OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        print(f"  Downloaded {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def save_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Save ticker data to CSV file.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol
    """
    ensure_data_dir()
    filepath = get_ticker_filepath(ticker)
    df.to_csv(filepath)
    print(f"  Saved to {filepath}")


def load_data(ticker: str) -> pd.DataFrame:
    """
    Load ticker data from CSV file.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        DataFrame with OHLCV data, or None if file doesn't exist
    """
    filepath = get_ticker_filepath(ticker)
    
    if not os.path.exists(filepath):
        print(f"No local data found for {ticker}")
        return None
    
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    return df


def download_all_tickers(tickers: list = None, years: int = None, force: bool = False) -> dict:
    """
    Download data for all tickers.
    
    Args:
        tickers: List of ticker symbols (default from config)
        years: Number of years of historical data
        force: If True, re-download even if local file exists
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    if tickers is None:
        tickers = ALL_TICKERS
    
    ensure_data_dir()
    data = {}
    
    for ticker in tickers:
        # Check if local file exists
        if not force and os.path.exists(get_ticker_filepath(ticker)):
            print(f"Loading {ticker} from local file...")
            df = load_data(ticker)
            if df is not None:
                data[ticker] = df
                continue
        
        # Download new data
        df = download_ticker_data(ticker, years)
        if df is not None:
            save_data(df, ticker)
            data[ticker] = df
    
    return data


def get_data_summary(data: dict) -> pd.DataFrame:
    """
    Get summary statistics for downloaded data.
    
    Args:
        data: Dictionary mapping ticker to DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    summary = []
    
    for ticker, df in data.items():
        if df is not None:
            summary.append({
                'Ticker': ticker,
                'Start Date': df.index.min().strftime('%Y-%m-%d'),
                'End Date': df.index.max().strftime('%Y-%m-%d'),
                'Rows': len(df),
                'Missing Close': df['Close'].isna().sum()
            })
    
    return pd.DataFrame(summary)


def update_data(tickers: list = None) -> dict:
    """
    Update local data by downloading only new data since last update.
    
    Args:
        tickers: List of ticker symbols (default from config)
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    if tickers is None:
        tickers = ALL_TICKERS
    
    ensure_data_dir()
    data = {}
    
    for ticker in tickers:
        existing_df = load_data(ticker)
        
        if existing_df is not None:
            # Get the last date in existing data
            last_date = existing_df.index.max().strftime('%Y-%m-%d')
            
            # Download new data since last date
            new_df = download_ticker_data(ticker, start_date=last_date)
            
            if new_df is not None and len(new_df) > 0:
                # Combine and remove duplicates
                combined = pd.concat([existing_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                save_data(combined, ticker)
                data[ticker] = combined
            else:
                data[ticker] = existing_df
        else:
            # No existing data, download full history
            df = download_ticker_data(ticker)
            if df is not None:
                save_data(df, ticker)
                data[ticker] = df
    
    return data


if __name__ == "__main__":
    # Test data download
    print("=" * 50)
    print("Testing data download...")
    print("=" * 50)
    
    data = download_all_tickers(force=False)
    
    print("\n" + "=" * 50)
    print("Data Summary:")
    print("=" * 50)
    print(get_data_summary(data).to_string(index=False))
