import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fetch stock data
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        if data is None or data.empty:
            print(f"No data found for {ticker}")
            return None
        data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj_Close', 'Volume': 'Volume'}, inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to preprocess stock data (including adding technical indicators)
def preprocess_data(data, sequence_length=60):
    # Add technical indicators using TA-Lib
    data['SMA_2'] = data['Close'].rolling(window=2).mean()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()

    # Bollinger Bands
    data['bb_bbm'], data['bb_bbh'], data['bb_bbl'] = ta.BBANDS(data['Close'], timeperiod=20)

    # MACD
    data['macd'], data['macd_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Average True Range
    data['atr'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Exponential Moving Averages
    data['ema_10'] = ta.EMA(data['Close'], timeperiod=10)
    data['ema_50'] = ta.EMA(data['Close'], timeperiod=50)

    # Relative Strength Index (RSI)
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)

    # Additional indicators you might want to include
    data['stochastic'] = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)[0]
    data['SAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['CMF'] = ta.CMF(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=20)
    data['Williams_%R'] = ta.WILLIAMS(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Drop rows with NaN values created by rolling and technical indicator calculations
    data.dropna(inplace=True)

    # Features to use for the model
    features = ['Close', 'Volume', 'SMA_2', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_30',
                'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'macd_signal', 'atr', 'ema_10', 
                'ema_50', 'RSI', 'stochastic', 'SAR', 'OBV', 'ADX', 'CMF', 'Williams_%R']

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # Prepare the dataset for training
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i])  # Last 'sequence_length' days
        y.append(scaled_data[i, 0])  # Predict 'Close' price

    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))  # Reshaping for LSTM input

    return x, y, scaler, data.index

# LSTM Model Creation
def create_and_train_model(x_train, y_train):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model

# Making predictions with the trained LSTM model
def make_prediction(model, last_sequence, scaler):
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted_price = model.predict(last_sequence, verbose=0)
    prediction = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], last_sequence.shape[2] - 1))), axis=1))
    return prediction[0, 0]

# Chatbot function for Gradio interface
def chatbot(messages):
    ticker = messages.strip().upper()

    # Check if the ticker is valid
    if not ticker.isalpha():
        return "Please enter a valid stock ticker (e.g., AAPL, MSFT)."

    stock_data = get_stock_data(ticker)

    if stock_data is not None:
        # Preprocess data
        x, y, scaler, index = preprocess_data(stock_data)

        # Split into training data
        split_index = int(len(x) * 0.8)
        x_train, y_train = x[:split_index], y[:split_index]

        # Train LSTM model
        model = create_and_train_model(x_train, y_train)

        # Use the last 60 days data to make prediction
        last_60_days = stock_data[['Close', 'Volume', 'SMA_2', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'macd_signal', 'atr', 'ema_10', 'ema_50', 'RSI', 'stochastic', 'SAR', 'OBV', 'ADX', 'CMF', 'Williams_%R']].iloc[-60:].values

        predicted_price = make_prediction(model, last_60_days, scaler)
        last_date = stock_data.index[-1]
        prediction_date = last_date + pd.Timedelta(days=1)

        return f"Predicted price for {ticker} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_price:.2f}"
    else:
        return f"Could not retrieve data for ticker: {ticker}. Please check the ticker symbol."

# Create a Gradio chatbot interface
import gradio as gr
iface = gr.Interface(
    fn=chatbot,  # Function to be called when the user sends a message
    inputs=gr.Textbox(label="Enter Stock Ticker"),  # Input component for the user to enter stock ticker
    outputs=gr.Textbox(label="Prediction Result"),  # Output component to display the prediction
    title="Stock Price Prediction Chatbot",  # Title of the chatbot
    description="Enter a stock ticker (e.g., AAPL, MSFT) to get the next-day price prediction.",  # Description
)

# Launch the Gradio interface
iface.launch(share=True)
