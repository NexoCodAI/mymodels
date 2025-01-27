import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import ta  # For technical indicators
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance and clean up columns."""
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            print(f"No data found for {ticker}")
            return None

        # Check for multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)  # Use the second level of MultiIndex

        # Rename columns for clarity
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

        print("Cleaned data columns:", data.columns)  # Debug flattened columns
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def preprocess_data(data):
    """Preprocess stock data by adding technical indicators and scaling."""
    if data is None or data.empty:
        print("Data is empty or None.")
        return None, None, None, None

    # Ensure 'Close' is numeric
    try:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'Close' to numeric: {e}")
        return None, None, None, None

    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    try:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        data['RSI'] = np.nan
        data['MACD'] = np.nan

    data.dropna(inplace=True)

    # Scaling
    features = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # Create sequences for LSTM
    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'Close' price
    x, y = np.array(x), np.array(y)
    return x, y, scaler, data.index

# Main script
ticker = "QNC.V"
stock_data = get_stock_data(ticker)

if stock_data is not None:
    print("Fetched and cleaned stock data:")
    print(stock_data.head())  # Debug fetched data

    x, y, scaler, index = preprocess_data(stock_data)
    if x is not None:
        print("Preprocessing successful.")
        print(f"x shape: {x.shape}, y shape: {y.shape}")

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))  # Output layer for predicting the 'Close' price

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x, y, batch_size=32, epochs=50)

      # Assuming you have already trained your model and have the scaler
# Get the last 60 days of data
last_60_days = stock_data[-60:].copy()

# Ensure 'Close' is numeric
last_60_days['Close'] = pd.to_numeric(last_60_days['Close'], errors='coerce')

# Feature Engineering for the last 60 days
last_60_days['SMA_10'] = last_60_days['Close'].rolling(window=10).mean()
last_60_days['SMA_20'] = last_60_days['Close'].rolling(window=20).mean()
last_60_days['RSI'] = ta.momentum.RSIIndicator(last_60_days['Close']).rsi()
last_60_days['MACD'] = ta.trend.MACD(last_60_days['Close']).macd()

# Drop NaN values
last_60_days.dropna(inplace=True)

# Scale the features
features = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD']
scaled_last_60 = scaler.transform(last_60_days[features])

# Prepare the input for the model
X_test = scaled_last_60.reshape((1, scaled_last_60.shape[0], scaled_last_60.shape[1]))

# Make the prediction
predicted_price = model.predict(X_test)

# Inverse transform the predicted price
predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], 5))), axis=1))[:, 0]

# Print the predicted price for tomorrow
print(f"Predicted Close Price for Tomorrow: {predicted_price[0]}")