import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(
    page_title="Hybrid LSTM-ARIMA Forecasting",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

# Set page configuration
st.set_page_config(page_title="Hybrid LSTM-ARIMA Forecasting", layout="wide")

# Page title
st.title("Hybrid Financial Time Series Forecasting")
st.markdown("### LSTM + ARIMA Model for S&P 500 Prediction")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Data parameters
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=1500))
end_date = st.sidebar.date_input("End Date", datetime.now())
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# LSTM parameters
st.sidebar.subheader("LSTM Parameters")
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 64)
lstm_dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
lstm_epochs = st.sidebar.slider("Epochs", 10, 100, 50)
sequence_length = st.sidebar.slider("Sequence Length (Days)", 10, 60, 30)

# ARIMA parameters
st.sidebar.subheader("ARIMA Parameters")
p = st.sidebar.slider("p (AR order)", 0, 5, 2)
d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
q = st.sidebar.slider("q (MA order)", 0, 5, 2)

# Function to load data
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to prepare data for LSTM
def prepare_lstm_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Determine train-test split (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Function to build LSTM model
def build_lstm_model(seq_length, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train ARIMA model
def train_arima_model(data, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

# Function to create hybrid predictions
def hybrid_forecast(arima_pred, lstm_pred, weights=(0.5, 0.5)):
    return weights[0] * arima_pred + weights[1] * lstm_pred

# Main app
data = load_data(ticker, start_date, end_date)

if data is not None:
    # Display raw data
    st.subheader("Raw Data Preview")
    st.dataframe(data.head())
    
    # Feature selection
    st.subheader("Feature Selection")
    available_columns = data.columns.tolist()
    
    # Fix for the error - safely find a default feature
    default_features = ['Adj Close', 'Close']
    default_feature = next((feat for feat in default_features if feat in available_columns), available_columns[0])
    
    # Get the index of the default feature safely
    default_index = available_columns.index(default_feature) if default_feature in available_columns else 0
    
    feature = st.selectbox("Select price feature to forecast", 
                      available_columns,
                      index=default_index)
    
    # Display time series chart
    st.subheader("Historical Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[feature], mode='lines', name=f'{ticker} {feature}'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Prepare data for models
    price_data = data[feature].values
    
    # For LSTM
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(price_data, sequence_length)
    
    # Train test split for both models (using same split points)
    train_size = int(len(price_data) * 0.8)
    train_data, test_data = price_data[:train_size], price_data[train_size:]
    
    # Creating index for plotting
    train_dates = data.index[:train_size]
    test_dates = data.index[train_size:]
    
    if st.button("Train Models and Forecast"):
        with st.spinner("Training LSTM model..."):
            # Build and train LSTM model
            lstm_model = build_lstm_model(sequence_length, lstm_units, lstm_dropout)
            history = lstm_model.fit(
                X_train, y_train,
                epochs=lstm_epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Make LSTM predictions
            lstm_test_pred = lstm_model.predict(X_test)
            lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
            
        with st.spinner("Training ARIMA model..."):
            # Train ARIMA model
            arima_model = train_arima_model(train_data, p, d, q)
            
            # Make ARIMA predictions
            arima_forecast = arima_model.forecast(steps=len(test_data))
            
        # Hybrid model - weight optimization
        best_rmse = float('inf')
        best_weights = (0.5, 0.5)
        
        weight_options = [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.5, 0.5), 
                          (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)]
        
        weight_results = []
        
        for w in weight_options:
            # Align predictions and ground truth to same length #change
            arima_len = len(arima_forecast)
            lstm_len = len(lstm_test_pred.flatten())
            test_len = len(test_data[sequence_length:])
            min_len = min(arima_len, lstm_len, test_len) #change
            arima_pred = np.array(arima_forecast[:min_len]) #change
            lstm_pred = np.array(lstm_test_pred.flatten()[:min_len]) #change
            test_data_aligned = test_data[sequence_length:sequence_length+min_len] #change
            hybrid_pred = hybrid_forecast(arima_pred, lstm_pred, weights=w) #change
            # Calculate RMSE #change
            rmse = math.sqrt(mean_squared_error(test_data_aligned, hybrid_pred)) #change
            weight_results.append({
                'ARIMA Weight': w[0],
                'LSTM Weight': w[1],
                'RMSE': rmse
            })
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = w
        
        # Create final hybrid predictions with best weights #change
        arima_len = len(arima_forecast)
        lstm_len = len(lstm_test_pred.flatten())
        test_len = len(test_data[sequence_length:])
        min_len = min(arima_len, lstm_len, test_len) #change
        arima_pred = np.array(arima_forecast[:min_len]) #change
        lstm_pred = np.array(lstm_test_pred.flatten()[:min_len]) #change
        test_data_aligned = test_data[sequence_length:sequence_length+min_len] #change
        hybrid_pred = hybrid_forecast(arima_pred, lstm_pred, weights=best_weights) #change
        # Calculate metrics #change
        lstm_rmse = math.sqrt(mean_squared_error(test_data_aligned, lstm_pred)) #change
        arima_rmse = math.sqrt(mean_squared_error(test_data[:min_len], arima_pred)) #change
        hybrid_rmse = math.sqrt(mean_squared_error(test_data_aligned, hybrid_pred)) #change
        
        # Calculate improvement percentage
        baseline_rmse = min(lstm_rmse, arima_rmse)
        improvement = ((baseline_rmse - hybrid_rmse) / baseline_rmse) * 100
        
        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LSTM RMSE", f"{lstm_rmse:.2f}")
        col2.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
        col3.metric("Hybrid RMSE", f"{hybrid_rmse:.2f}")
        col4.metric("Improvement", f"{improvement:.2f}%")
        
        # Display optimal weights
        st.subheader("Optimal Model Weights")
        st.write(f"ARIMA Weight: {best_weights[0]:.2f}, LSTM Weight: {best_weights[1]:.2f}")
        
        # Display weights comparison table
        st.subheader("Weight Optimization Results")
        st.dataframe(pd.DataFrame(weight_results))
        
        # Plot results
        st.subheader("Model Predictions")
        
        # Create plotly figure
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=("Price Predictions", "Prediction Error"))
        
        # Add actual values
        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:], 
                      y=test_data[sequence_length:],
                      mode='lines',
                      name='Actual',
                      line=dict(color='black')),
            row=1, col=1
        )
        
        # Add LSTM predictions
        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:], 
                      y=lstm_test_pred.flatten(),
                      mode='lines',
                      name='LSTM',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add ARIMA predictions
        fig.add_trace(
            go.Scatter(x=test_dates, 
                      y=arima_forecast,
                      mode='lines',
                      name='ARIMA',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Add Hybrid predictions
        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:], 
                      y=hybrid_pred,
                      mode='lines',
                      name='Hybrid',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Add error plots
        # Align error plot arrays to same length #change
        error_lstm_len = min(len(test_data[sequence_length:]), len(lstm_test_pred.flatten())) #change
        error_arima_len = min(len(test_data[sequence_length:]), len(arima_forecast[sequence_length:])) #change
        error_hybrid_len = min(len(test_data[sequence_length:]), len(hybrid_pred)) #change

        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:sequence_length+error_lstm_len], 
                      y=np.abs(test_data[sequence_length:sequence_length+error_lstm_len] - lstm_test_pred.flatten()[:error_lstm_len]), #change
                      mode='lines',
                      name='LSTM Error',
                      line=dict(color='blue')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:sequence_length+error_arima_len], 
                      y=np.abs(test_data[sequence_length:sequence_length+error_arima_len] - arima_forecast[sequence_length:sequence_length+error_arima_len]), #change
                      mode='lines',
                      name='ARIMA Error',
                      line=dict(color='green')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=test_dates[sequence_length:sequence_length+error_hybrid_len], 
                      y=np.abs(test_data[sequence_length:sequence_length+error_hybrid_len] - hybrid_pred[:error_hybrid_len]), #change
                      mode='lines',
                      name='Hybrid Error',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=800, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
        # Future forecasting
        st.subheader(f"Future {forecast_days} Days Forecast")
        
        # Prepare data for future forecasting
        last_sequence = price_data[-sequence_length:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # LSTM future forecast
        X_future = last_sequence_scaled.reshape(1, sequence_length, 1)
        lstm_future_pred = []
        
        current_batch = X_future
        for i in range(forecast_days):
            # Get prediction for next day
            current_pred = lstm_model.predict(current_batch)[0]
            lstm_future_pred.append(current_pred[0])
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], 
                                     [[current_pred]], 
                                     axis=1)
        
        # Convert LSTM predictions back to original scale
        lstm_future_pred = scaler.inverse_transform(np.array(lstm_future_pred).reshape(-1, 1))
        
        # ARIMA future forecast
        arima_future = arima_model.forecast(steps=forecast_days)
        
        # Create hybrid future forecast
        hybrid_future = hybrid_forecast(arima_future, lstm_future_pred.flatten(), weights=best_weights)
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Create dataframe for forecast results
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM Forecast': lstm_future_pred.flatten(),
            'ARIMA Forecast': arima_future,
            'Hybrid Forecast': hybrid_future
        })
        
        # Display forecast dataframe
        st.dataframe(forecast_df.set_index('Date'))
        
        # Plot future forecast
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(x=data.index, 
                               y=data[feature],
                               mode='lines',
                               name='Historical',
                               line=dict(color='black')))
        
        # Add LSTM forecast
        fig.add_trace(go.Scatter(x=future_dates, 
                               y=lstm_future_pred.flatten(),
                               mode='lines',
                               name='LSTM Forecast',
                               line=dict(color='blue')))
        
        # Add ARIMA forecast
        fig.add_trace(go.Scatter(x=future_dates, 
                               y=arima_future,
                               mode='lines',
                               name='ARIMA Forecast',
                               line=dict(color='green')))
        
        # Add Hybrid forecast
        fig.add_trace(go.Scatter(x=future_dates, 
                               y=hybrid_future,
                               mode='lines',
                               name='Hybrid Forecast',
                               line=dict(color='red')))
        
        # Add vertical line to separate historical data and forecasts
        fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="gray")
        
        fig.update_layout(title=f"{ticker} {feature} Forecast for Next {forecast_days} Days",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend_title="Models")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download option for forecast
        st.download_button(
            label="Download Forecast Data",
            data=forecast_df.to_csv(index=False),
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )