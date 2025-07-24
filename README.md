# Hybrid-Lstm-stock-prediction
Hybrid LSTM-ARIMA Forecasting is a Streamlit web app for financial time series forecasting. It combines LSTM (deep learning) and ARIMA (statistical) models to predict stock prices, visualize results, and optimize hybrid model weights for improved accuracy.

# Hybrid LSTM-ARIMA Financial Time Series Forecasting

This project is a Streamlit web application for financial time series forecasting using a hybrid approach that combines LSTM (Long Short-Term Memory) neural networks and ARIMA (AutoRegressive Integrated Moving Average) models. The app allows users to forecast stock prices (e.g., S&P 500) by leveraging the strengths of both deep learning and statistical methods, visualize results, and optimize model weights for best performance.

## Features

- Interactive Streamlit web interface
- Download historical stock data from Yahoo Finance
- Forecast using LSTM, ARIMA, and a hybrid of both
- Visualize historical data, predictions, and errors
- Optimize hybrid model weights for best RMSE
- Download forecast results as CSV

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hybrid-lstm-arima.git
   cd hybrid-lstm-arima
   ```

2. **(Recommended) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```



3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

To start the Streamlit app, run:

```bash
streamlit run Hybrid.py
```

The app will open in your browser. You can select the ticker symbol, date range, model parameters, and forecast horizon from the sidebar.

### Usage

- Enter the stock ticker (e.g., `SPY` for S&P 500).
- Select the date range and model parameters in the sidebar.
- Click **"Train Models and Forecast"** to run the models.
- View performance metrics, prediction plots, and future forecasts.
- Download the forecast data as a CSV file.

## Dependencies

- streamlit
- pandas
- numpy
- yfinance
- matplotlib
- statsmodels
- scikit-learn
- tensorflow
- plotly

(See `requirements.txt` for the full list.)

## License

[Specify your license here, e.g., MIT]

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [Streamlit](https://streamlit.io/) for the web app framework

---

*Feel free to open issues or pull requests for improvements!*
