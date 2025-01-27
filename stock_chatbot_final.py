import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
import gradio as gr
import talib as ta
import ta as ta_lib
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Attention, Input
from tensorflow.keras.models import Model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_stock_data(ticker, period="5d", interval="1m"):
    """Fetch stock data using yfinance and clean up columns."""
    try:
        # Fetch stock data using yfinance
        data = yf.download(ticker, period=period, interval = interval)

        if data is None or data.empty:
            print(f"No data found for {ticker}")
            return None

        # If the DataFrame has a MultiIndex for columns, flatten it
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex by selecting the second level (e.g., the actual column names)
            data.columns = data.columns.get_level_values(0)

        # Rename columns for consistency
        data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        }, inplace=True)

        # Check for required columns
        required_columns = ['Close', 'Volume', 'High', 'Low']
        if not all(col in data.columns for col in required_columns):
            print(f"Data is missing required columns: {required_columns}")
            return None

        print("Cleaned data columns:", data.columns)  # Debug
        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_news_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(article)['compound'] for article in news]
    return np.mean(sentiment_scores)

ticker = "AAPL"
stock_data = get_stock_data(ticker)
# Step 2: Fetch News Data (example list or from an API)
news = []
# Step 3: Calculate Sentiment Scores
sentiment_score = get_news_sentiment(news)
print(f"Sentiment Score: {sentiment_score}")

sentiment_scores = [sentiment_score] * len(stock_data)  # Example: Repeat for simplicity
stock_data['sentiment'] = sentiment_scores


# Function to get latest close price for a given ticker
def get_latest_price(ticker):
    data = yf.download(ticker, period="1d", interval="1m")
    latest_price = data['Close'].iloc[-1]  # Get the last available closing price
    return latest_price

ticker = 'AAPL'  # Example ticker
latest_price = get_latest_price(ticker)
print(f"Latest Close Price: {latest_price}")



def preprocess_data(data, sentiment_scores=None):
    """Preprocess stock data by adding technical indicators and scaling."""
    if data is None or data.empty:
        print("Data is empty or None.")
        return None, None, None, None

    # Check for required columns
    required_columns = ['Close', 'Volume', 'High', 'Low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Data is missing required columns: {missing_columns}")
        return None, None, None, None

    # Feature Engineering
    data['SMA_2'] = data['Close'].rolling(window=2).mean()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    
    # Add sentiment scores if provided
    if sentiment_scores is not None and len(sentiment_scores) == len(data):
        data['sentiment'] = sentiment_scores
    else:
        # Default to 0 sentiment if not provided
        data['sentiment'] = 0
    
     # Add percent change from last close
    data['pct_change'] = data['Close'].pct_change()

    # Add rolling volatility
    data['volatility'] = data['Close'].rolling(window=10).std()

    # Add recent price momentum
    data['momentum'] = data['Close'] - data['Close'].shift(10)
    

    # Bollinger Bands
    data['bb_bbm'], data['bb_bbh'], data['bb_bbl'] = ta.BBANDS(data['Close'], timeperiod=20)

    # MACD
    data['macd'], data['macd_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # ATR
    data['atr'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # EMA
    data['ema_10'] = ta.EMA(data['Close'], timeperiod=10)
    data['ema_50'] = ta.EMA(data['Close'], timeperiod=50)
    data['ema_5'] = ta.EMA(data['Close'], timeperiod=5)

    # Calculate RSI (Relative Strength Index)
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    
        # Calculate Stochastic Oscillator
    data['stochastic'], _ = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3)

    # Calculate Williams %R
    data['Williams_%R'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Calculate On-Balance Volume (OBV)
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])

    # Calculate Chaikin Money Flow (CMF)
    data['CMF'] = ta_lib.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'], window=20)

    # Calculate Average Directional Index (ADX)
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Calculate Parabolic SAR
    data['SAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    if sentiment_scores is not None:
        data['sentiment'] = sentiment_scores
        
    # Drop NaN rows
    data.dropna(inplace=True)

    # Scaling
    features = ['Close', 'Volume', 'volatility', 'pct_change', 'momentum', 
                'stochastic', 'SAR', 'OBV', 'ADX', 
                'CMF', 'Williams_%R', 'SMA_2', 'SMA_5', 'SMA_10', 'SMA_20',
                'SMA_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 
                'macd_signal', 'atr', 'ema_10', 'ema_50', 'ema_5', 'RSI', 'sentiment']
    
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
'''
def weighted_mse(y_true, y_pred):
  # Assuming your weights are a 1D array with length equal to total data points
  weights = ...  # Load or calculate your weights

  # Ensure weights have the same batch size as y_true
  batch_size = K.shape(y_true)[0]
  weights_repeated = K.repeat(K.expand_dims(weights, axis=0), repeats=batch_size)

  loss = K.square(y_true - y_pred) * K.expand_dims(weights_repeated, axis=-1)
  return loss
'''

from tensorflow.keras.layers import GRU, Attention, Layer
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Sequential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define the model creation function
def create_model_with_attention(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    gru_out = GRU(64, return_sequences=True)(inputs)
    attention_out = Attention()([gru_out, gru_out])
    dense_out = Dense(1)(attention_out)
    model = Model(inputs, dense_out)
    model.compile(optimizer='adam', loss='mse')
    return model

# Train and validate the model
def create_and_train_model(x_train, y_train, timesteps, features):
    model = create_model_with_attention(timesteps, features)

    # Learning rate scheduler and early stopping
    lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * (0.5 ** (epoch // 10)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        epochs=35,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[lr_schedule, early_stopping]
    )
    return model, history

# Prediction function
def make_prediction(model, last_sequence, scaler):
    features = ['Close', 'Volume', 'volatility', 'pct_change', 'momentum', 
                'stochastic', 'SAR', 'OBV', 'ADX', 
                'CMF', 'Williams_%R', 'SMA_2', 'SMA_5', 'SMA_10', 'SMA_20',
                'SMA_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 
                'macd_signal', 'atr', 'ema_10', 'ema_50', 'ema_5', 'RSI', 'sentiment' ]
    last_sequence_df = pd.DataFrame(last_sequence, columns=features)

    # Scale the sequence
    scaled_last_sequence = scaler.transform(last_sequence_df)

    # Reshape for model input
    scaled_last_sequence = scaled_last_sequence.reshape(1, scaled_last_sequence.shape[0], scaled_last_sequence.shape[1])

    # Predict
    prediction = model.predict(scaled_last_sequence, verbose=0)

    # Ensure prediction is 2D
    prediction = prediction.reshape(-1, 1)

    # Create zeros_array for padding
    zeros_array = np.zeros((prediction.shape[0], len(features) - 1))

    # Concatenate for inverse transform
    concatenated = np.concatenate((prediction, zeros_array), axis=1)

    # Inverse transform and keep the first column
    prediction = scaler.inverse_transform(concatenated)[:, 0]

    return prediction[0]

# Predict stock price for a given ticker
def predict_stock(ticker):
    stock_data = get_stock_data(ticker)

    if stock_data is not None:
        # Example news data for sentiment analysis
        news = ["Apple launches new iPhone with groundbreaking features.",
            "Economic slowdown could affect tech sector."]
        sentiment_score = get_news_sentiment(news)
        sentiment_scores = [sentiment_score] * len(stock_data)
        
        # Preprocess data first to generate x, y, scaler, and index
        x, y, scaler, index = preprocess_data(stock_data, sentiment_scores)
        
        
        if x is not None:
            split_index = int(len(x) * 0.8)
            x_train, x_test = x[:split_index], x[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Get timesteps and features
            timesteps = x.shape[1]
            features = x.shape[2]

            model, _ = create_and_train_model(x_train, y_train, timesteps, features)

            required_features = ['Close', 'Volume', 'volatility', 'pct_change', 
                                 'momentum', 'stochastic', 'SAR', 'OBV', 'ADX', 
                'CMF', 'Williams_%R', 'SMA_2', 'SMA_5', 'SMA_10', 'SMA_20',
                'SMA_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 
                'macd_signal', 'atr', 'ema_10', 'ema_50', 'ema_5', 'RSI', 'sentiment' ]
            
            if not all(f in stock_data.columns for f in required_features):
                return "Not all features are present in the stock data. Prediction cannot be made."

            last_60_days = stock_data[required_features].iloc[-60:].values
            predicted_price = make_prediction(model, last_60_days, scaler)
            last_date = stock_data.index[-1]
            prediction_date = last_date + pd.Timedelta(days=1)

            return f"Predicted price for {ticker} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_price:.2f}"
        else:
            return "Not enough data to make a prediction for this ticker."
    else:
        return f"Could not retrieve data for ticker: {ticker}. Please check the ticker symbol."


# Gradio Chat Interface
def chatbot(messages, *args):
    """Chatbot interface for stock price predictions."""
    try:
        ticker = messages.strip().upper()
        if not ticker.isalpha():
            return "Please enter a valid stock ticker (e.g., AAPL, MSFT)."
        return predict_stock(ticker)
    except Exception as e:
        return f"An error occurred: {e}"


iface = gr.ChatInterface(
    fn=chatbot,
    title="Stock Price Predictor Chatbot",
    description="Enter a stock ticker (e.g., AAPL, MSFT) to get a next-day price prediction.",
)
iface.launch(share=True)
