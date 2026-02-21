"""
Streamlit dashboard for LSTM Portfolio Risk Assistant.
Includes inference functions and portfolio aggregation.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import torch

from config import (
    ALL_TICKERS,
    TICKERS,
    REGIME_CLASSES,
    REGIME_COLORS,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON,
    DEFAULT_WEIGHTS
)
from data import load_data, download_all_tickers
from features import build_features, get_latest_sequence
from model import build_lstm_model


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler (cached)."""
    try:
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        
        # For PyTorch, we need to know the input shape
        # We'll load a sample to get it
        sample_data = load_data(ALL_TICKERS[0])
        if sample_data is None:
            return None, None
        
        sample_features = build_features(sample_data)
        n_features = len([col for col in sample_features.columns if col not in ['regime_label', 'ticker']])
        
        # Build model with correct input shape
        input_shape = (SEQUENCE_LENGTH, n_features)
        model = build_lstm_model(input_shape=input_shape)
        
        # Load weights
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        
        return model, scaler
    except FileNotFoundError as e:
        return None, None


def predict_regime(model, sequence: np.ndarray) -> dict:
    """
    Get regime probabilities for a single ticker.
    
    Args:
        model: Trained LSTM model
        sequence: Input sequence (1, seq_length, n_features)
    
    Returns:
        Dictionary with regime and probabilities
    """
    if sequence is None:
        return None
    
    # Get probabilities
    with torch.no_grad():
        X_tensor = torch.FloatTensor(sequence)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()
    
    # Get predicted regime
    regime_idx = np.argmax(probabilities)
    regime = REGIME_CLASSES[regime_idx]
    confidence = probabilities[regime_idx]
    
    return {
        'regime': regime,
        'confidence': float(confidence),
        'probabilities': {
            REGIME_CLASSES[i]: float(probabilities[i]) 
            for i in range(len(REGIME_CLASSES))
        }
    }


def predict_all_tickers(model, scaler, tickers: list = None) -> dict:
    """
    Predict regime for all tickers.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted StandardScaler
        tickers: List of tickers (default from config)
    
    Returns:
        Dictionary mapping ticker to prediction
    """
    if tickers is None:
        tickers = ALL_TICKERS
    
    predictions = {}
    
    for ticker in tickers:
        sequence = get_latest_sequence(ticker, scaler)
        
        if sequence is not None:
            predictions[ticker] = predict_regime(model, sequence)
        else:
            predictions[ticker] = None
    
    return predictions


def compute_portfolio_risk(predictions: dict, weights: dict) -> dict:
    """
    Compute aggregate portfolio risk.
    
    Args:
        predictions: Dictionary of ticker predictions
        weights: Dictionary of portfolio weights
    
    Returns:
        Dictionary with portfolio risk metrics
    """
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return None
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Compute weighted high-risk probability
    weighted_high_risk = 0.0
    high_risk_weight = 0.0
    
    for ticker, pred in predictions.items():
        if pred is not None and ticker in normalized_weights:
            weight = normalized_weights[ticker]
            high_prob = pred['probabilities']['high']
            weighted_high_risk += weight * high_prob
            
            if pred['regime'] == 'high':
                high_risk_weight += weight
    
    # Determine portfolio regime
    if weighted_high_risk > 0.5:
        portfolio_regime = 'high'
    elif weighted_high_risk > 0.25:
        portfolio_regime = 'medium'
    else:
        portfolio_regime = 'low'
    
    # Generate summary text
    if portfolio_regime == 'high':
        summary = f"Portfolio is in HIGH risk regime. {high_risk_weight:.0%} of portfolio weight is in assets flagged as high-risk for the next {PREDICTION_HORIZON} days. Consider reducing exposure."
    elif portfolio_regime == 'medium':
        summary = f"Portfolio is in MEDIUM risk regime. {high_risk_weight:.0%} of portfolio weight is in high-risk assets. Monitor closely."
    else:
        summary = f"Portfolio is in LOW risk regime. Only {high_risk_weight:.0%} of portfolio weight is in high-risk assets. Current outlook is stable."
    
    return {
        'portfolio_regime': portfolio_regime,
        'risk_score': weighted_high_risk,
        'high_risk_weight': high_risk_weight,
        'summary': summary
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_price_chart_with_regimes(ticker: str, df: pd.DataFrame, 
                                     regime_history: pd.DataFrame = None,
                                     days: int = 252) -> go.Figure:
    """
    Create price chart with regime background colors.
    
    Args:
        ticker: Ticker symbol
        df: OHLCV DataFrame
        regime_history: DataFrame with regime predictions (optional)
        days: Number of days to show
    
    Returns:
        Plotly figure
    """
    # Filter to last N days
    df_recent = df.tail(days).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2C3E50', width=2)
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#27AE60' if df_recent['Close'].iloc[i] >= df_recent['Open'].iloc[i] 
              else '#E74C3C' for i in range(len(df_recent))]
    
    fig.add_trace(
        go.Bar(
            x=df_recent.index,
            y=df_recent['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add regime background if available
    if regime_history is not None and len(regime_history) > 0:
        # This would require historical regime predictions
        # For now, we'll skip this feature
        pass
    
    fig.update_layout(
        title=f'{ticker} Price Chart',
        template='plotly_white',
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig


def create_regime_gauge(prediction: dict) -> go.Figure:
    """
    Create a gauge chart for regime probabilities.
    
    Args:
        prediction: Prediction dictionary with probabilities
    
    Returns:
        Plotly figure
    """
    if prediction is None:
        return None
    
    fig = go.Figure()
    
    # Create gauge for high-risk probability
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=prediction['probabilities']['high'] * 100,
        title={'text': 'High Risk Probability (%)'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': REGIME_COLORS['high']},
            'steps': [
                {'range': [0, 33], 'color': REGIME_COLORS['low']},
                {'range': [33, 67], 'color': REGIME_COLORS['medium']},
                {'range': [67, 100], 'color': REGIME_COLORS['high']}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': prediction['probabilities']['high'] * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_portfolio_risk_chart(predictions: dict, weights: dict) -> go.Figure:
    """
    Create a bar chart showing risk contribution by ticker.
    
    Args:
        predictions: Dictionary of predictions
        weights: Dictionary of weights
    
    Returns:
        Plotly figure
    """
    # Prepare data
    data = []
    total_weight = sum(weights.values()) or 1
    
    for ticker in ALL_TICKERS:
        if ticker in predictions and predictions[ticker] is not None:
            weight = weights.get(ticker, 0) / total_weight
            high_prob = predictions[ticker]['probabilities']['high']
            risk_contribution = weight * high_prob
            
            data.append({
                'Ticker': ticker,
                'Weight': weight * 100,
                'High Risk Prob': high_prob * 100,
                'Risk Contribution': risk_contribution * 100,
                'Regime': predictions[ticker]['regime']
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return None
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars for each regime
    for regime in REGIME_CLASSES:
        regime_df = df[df['Regime'] == regime]
        
        if len(regime_df) > 0:
            fig.add_trace(go.Bar(
                x=regime_df['Ticker'],
                y=regime_df['High Risk Prob'],
                name=regime.capitalize(),
                marker_color=REGIME_COLORS[regime],
                text=regime_df['High Risk Prob'].round(1).astype(str) + '%',
                textposition='outside'
            ))
    
    fig.update_layout(
        title='High Risk Probability by Ticker',
        template='plotly_white',
        height=400,
        showlegend=True,
        xaxis_title='Ticker',
        yaxis_title='High Risk Probability (%)',
        yaxis=dict(range=[0, 100])
    )
    
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title='LSTM Portfolio Risk Assistant',
        page_icon='📊',
        layout='wide'
    )
    
    # Title
    st.title('📊 LSTM Portfolio Risk Assistant')
    st.markdown(f'Predicting **{PREDICTION_HORIZON}-day volatility regime** for your portfolio')
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("""
        **Model not found!** 
        
        Please train the model first by running:
        ```
        python main.py
        ```
        
        This will download data, train the LSTM model, and save it to the `models/` directory.
        """)
        
        # Show option to download data only
        if st.button('Download Data Only (for testing)'):
            with st.spinner('Downloading data...'):
                download_all_tickers()
            st.success('Data downloaded! Now run `python main.py` to train the model.')
        
        return
    
    # Sidebar
    st.sidebar.header('Settings')
    
    # Tabs
    tab1, tab2 = st.tabs(['📈 Ticker View', '💼 Portfolio View'])
    
    # =========================================================================
    # Tab 1: Ticker View
    # =========================================================================
    with tab1:
        st.header('Ticker Risk Analysis')
        
        # Ticker selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Group tickers by type
            ticker_options = []
            ticker_options.extend([f"{t} (US ETF)" for t in TICKERS['us_etfs']])
            ticker_options.extend([f"{t} (MY Stock)" for t in TICKERS['my_stocks']])
            
            selected = st.selectbox('Select Ticker', ticker_options)
            ticker = selected.split(' ')[0]  # Extract ticker symbol
            
            # Chart period
            days = st.slider('Chart Period (days)', 30, 504, 252)
        
        # Get prediction
        with st.spinner('Getting prediction...'):
            sequence = get_latest_sequence(ticker, scaler)
            prediction = predict_regime(model, sequence)
        
        if prediction:
            # Display prediction
            with col1:
                st.markdown('### Current Regime Prediction')
                
                # Regime badge
                regime_color = REGIME_COLORS[prediction['regime']]
                st.markdown(
                    f"<h2 style='color: {regime_color};'>{prediction['regime'].upper()}</h2>",
                    unsafe_allow_html=True
                )
                
                st.metric('Confidence', f"{prediction['confidence']:.1%}")
                
                # Probability breakdown
                st.markdown('#### Probability Distribution')
                for regime, prob in prediction['probabilities'].items():
                    color = REGIME_COLORS[regime]
                    st.markdown(
                        f"<span style='color: {color};'>**{regime.capitalize()}**: {prob:.1%}</span>",
                        unsafe_allow_html=True
                    )
                
                # Gauge chart
                st.plotly_chart(create_regime_gauge(prediction), use_container_width=True)
            
            # Price chart
            with col2:
                df = load_data(ticker)
                if df is not None:
                    fig = create_price_chart_with_regimes(ticker, df, days=days)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f'Could not generate prediction for {ticker}. Check if data is available.')
    
    # =========================================================================
    # Tab 2: Portfolio View
    # =========================================================================
    with tab2:
        st.header('Portfolio Risk Summary')
        
        # Portfolio weights input
        st.markdown('### Adjust Portfolio Weights')
        
        weights = {}
        cols = st.columns(len(ALL_TICKERS))
        
        for i, ticker in enumerate(ALL_TICKERS):
            with cols[i]:
                default_weight = int(DEFAULT_WEIGHTS.get(ticker, 1/len(ALL_TICKERS)) * 100)
                weights[ticker] = st.slider(
                    ticker, 
                    0, 100, 
                    default_weight,
                    key=f'weight_{ticker}'
                )
        
        # Get all predictions
        with st.spinner('Analyzing portfolio...'):
            predictions = predict_all_tickers(model, scaler)
        
        # Compute portfolio risk
        portfolio_risk = compute_portfolio_risk(predictions, weights)
        
        if portfolio_risk:
            # Display portfolio summary
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('### Portfolio Risk Assessment')
                
                # Portfolio regime
                regime_color = REGIME_COLORS[portfolio_risk['portfolio_regime']]
                st.markdown(
                    f"<h2 style='color: {regime_color};'>{portfolio_risk['portfolio_regime'].upper()} RISK</h2>",
                    unsafe_allow_html=True
                )
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric('Risk Score', f"{portfolio_risk['risk_score']:.1%}")
                with col_b:
                    st.metric('High-Risk Weight', f"{portfolio_risk['high_risk_weight']:.1%}")
                
                # Summary
                st.info(portfolio_risk['summary'])
            
            with col2:
                # Risk contribution chart
                fig = create_portfolio_risk_chart(predictions, weights)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown('### Detailed Holdings')
            
            table_data = []
            total_weight = sum(weights.values()) or 1
            
            for ticker in ALL_TICKERS:
                if predictions.get(ticker) is not None:
                    pred = predictions[ticker]
                    weight = weights[ticker] / total_weight
                    
                    table_data.append({
                        'Ticker': ticker,
                        'Weight': f"{weight:.1%}",
                        'Regime': pred['regime'].capitalize(),
                        'Low Prob': f"{pred['probabilities']['low']:.1%}",
                        'Medium Prob': f"{pred['probabilities']['medium']:.1%}",
                        'High Prob': f"{pred['probabilities']['high']:.1%}",
                        'Risk Contribution': f"{weight * pred['probabilities']['high']:.2%}"
                    })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            # Regime color legend
            st.markdown('**Regime Legend:**')
            cols = st.columns(3)
            for i, regime in enumerate(REGIME_CLASSES):
                with cols[i]:
                    st.markdown(
                        f"<span style='color: {REGIME_COLORS[regime]};'>■</span> **{regime.capitalize()}**",
                        unsafe_allow_html=True
                    )
        else:
            st.warning('Could not compute portfolio risk. Check if predictions are available.')
    
    # Footer
    st.markdown('---')
    st.markdown(
        f"""
        <small>
        **Disclaimer**: This tool is for educational purposes only. 
        Predictions are based on historical patterns and may not reflect future market conditions.
        Model predicts {PREDICTION_HORIZON}-day forward volatility regime.
        </small>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
