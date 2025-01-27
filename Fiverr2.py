import pandas as pd  
import numpy as np  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, GRU, Attention, Layer  
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping  
import matplotlib.pyplot as plt  
import gradio as gr  
import talib as ta  
import ta as ta_lib  
import tensorflow.keras.backend as K  
from tensorflow.keras.regularizers import l2  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
import tensorflow as tf  
import yfinance as yf  
import math
import gc
import tracemalloc

# Start tracing memory allocations
tracemalloc.start()

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Clear session and garbage collection for memory management
tf.keras.backend.clear_session()
gc.collect()

# Function to fetch stock data
def get_stock_data(ticker, period="1y", interval="1d"):  
   try:  
      data = yf.download(ticker, period=period, interval=interval)  
      if data.empty:  
        print(f"No data found for {ticker} for the specified period and interval.")  
        return None  
      data = data.rename(columns={  
        'Open': 'Open',  
        'High': 'High',  
        'Low': 'Low',  
        'Close': 'Close',  
        'Adj Close': 'Adj_Close',  
        'Volume': 'Volume'  
      })  
      required_columns = ['Close', 'Volume', 'High', 'Low']  
      if not all(col in data.columns for col in required_columns):  
        print(f"Data for {ticker} is missing required columns.")  
        return None  
      data.reset_index(inplace=True)  
      data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']  
      return data  
   except Exception as e:  
      print(f"Error fetching data for {ticker}: {e}")  
      return None

# Function to get news sentiment
def get_news_sentiment(news):  
   analyzer = SentimentIntensityAnalyzer()  
   if not news:  
      return 0.0  
   try:  
      sentiment_scores = [analyzer.polarity_scores(article)['compound'] for article in news]  
      return np.mean(sentiment_scores)  
   except Exception as e:  
      print(f"Error calculating sentiment: {e}")  
      return 0.0  

def preprocess_data(data, sentiment_scores=None, lookback=40):
    if data is None or data.empty or len(data) < lookback + 1:
        print("Data is empty, None, or too short. Using available data.")
        lookback = len(data) - 1
        if lookback < 1:
            return None, None, None

    try:
        # Apply TA-Lib indicators
        for col in ['Close', 'High', 'Low', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        original_len = len(data)
        data.dropna(inplace=True)
        cleaned_len = len(data)
        if cleaned_len < lookback + 1:
            print(f"Not enough data after cleanup (minimum {lookback + 1} rows required). Original length: {original_len}, Cleaned length: {cleaned_len}")
            return None, None, None

        # Applying TA-Lib indicators
        data['bb_bbm'], data['bb_bbh'], data['bb_bbl'] = ta.BBANDS(data['Close'], timeperiod=20)
        data['macd'], data['macd_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['atr'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['ema_10'] = ta.EMA(data['Close'], timeperiod=10)
        data['ema_50'] = ta.EMA(data['Close'], timeperiod=50)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        data['stochastic'], _ = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3)
        data['OBV'] = ta.OBV(data['Close'], data['Volume'])
        data['CMF'] = ta_lib.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'], window=20)
        data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['SAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
        data['ewma_close'] = data['Close'].ewm(span=10).mean()
        data['SMA_2'] = data['Close'].rolling(window=2).mean()
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['pct_change'] = data['Close'].pct_change()
        data['volatility'] = data['Close'].rolling(window=10).std()
        data['momentum'] = data['Close'] - data['Close'].shift(10)
        data = data.ffill().bfill()  # Fill NaNs

        if sentiment_scores is not None:
            data['sentiment'] = sentiment_scores
        else:
            data['sentiment'] = 0.0

        features = ['Close', 'Volume', 'ewma_close', 'volatility', 'pct_change', 'momentum',
                    'stochastic', 'SAR', 'OBV', 'ADX', 'CMF', 'RSI', 'sentiment']
        data = data[features].copy()  # Select features before scaling to avoid errors
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        x, y = [], []
        for i in range(lookback, len(scaled_data)):
            x.append(scaled_data[i-lookback:i])  # Sequence of 'lookback' time steps
            y.append(scaled_data[i, 0])  # Using 'Close' as the target (index 0)
        x, y = np.array(x), np.array(y)
        return x, y, scaler
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None

# Custom attention layer with reduced memory usage
class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, inputs):
        query, value = inputs
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)
        attention_scores = tf.matmul(query, value, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(d_k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tracemalloc
import pandas as pd  # Assuming you are working with Pandas DataFrames for time series data

# Custom GRU model creation function
def create_gru_model(input_shape):
    """
    Create and compile a GRU-based model.

    Args:
        input_shape: Tuple representing the shape of the input (lookback, num_features).
    """
    inputs = Input(shape=input_shape)  # Expecting a tuple (lookback, num_features)
    x = GRU(64, return_sequences=True)(inputs)
    x = GRU(32)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Example usage with time-series data (e.g., x_train shape)
x_train = np.random.rand(100, 30, 26)  # Example shape (100 samples, 30 timesteps, 26 features)

# Correctly pass the input shape tuple
model = create_gru_model((x_train.shape[1], x_train.shape[2]))  # (timesteps, num_features)
model.summary()

# Display memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:  # Display top 10 memory usage locations
    print(stat)

# Example preprocessing function (this needs to be defined as per your dataset)
def preprocess_data(data, lookback):
    # Placeholder for actual preprocessing logic, returns x_data, y_data, scaler
    # Example: Returns a tuple of X and Y after processing data (e.g., normalization)
    return np.random.rand(len(data) - lookback, lookback, data.shape[1]), np.random.rand(len(data) - lookback), None

def walk_forward_validation(data, lookback, splits):
    """
    Performs walk-forward validation to evaluate a time series model.

    Args:
        data: Pandas DataFrame containing the time series data.
        lookback: The number of time steps to look back for each prediction.
        splits: The number of folds for cross-validation.

    Returns:
        A tuple containing:
            - The average validation loss across all folds.
            - The best performing model (the one with the lowest validation loss).
    """
    best_model = None
    min_val_loss = float('inf')
    results = []
    fold_size = len(data) // splits

    for i in range(splits):
        print(f"Processing fold {i + 1}...")
        start = i * fold_size
        end = start + fold_size if i < splits - 1 else len(data)

        train_data = data.iloc[:start].copy()
        val_data = data.iloc[start:end].copy()

        if train_data.empty or val_data.empty:
            print(f"Skipping fold {i + 1}: Empty train or validation data.")
            continue

        x_train, y_train, _ = preprocess_data(train_data, lookback=lookback)
        x_val, y_val, _ = preprocess_data(val_data, lookback=lookback)

        if x_train is None or len(x_train) == 0 or x_val is None or len(x_val) == 0:
            print(f"Skipping fold {i + 1} due to insufficient training or validation data.")
            continue

        model = create_gru_model((x_train.shape[1], x_train.shape[2]))  # Correct shape passed here
        lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * (0.5 ** (epoch // 10)))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            x_train, y_train,
            epochs=20, batch_size=3,
            validation_data=(x_val, y_val),
            verbose=1,
            callbacks=[lr_schedule, early_stopping]
        )

        val_loss = model.evaluate(x_val, y_val, verbose=0)
        results.append(val_loss)
        print(f"Fold {i + 1}: Validation Loss = {val_loss}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model

    if not results:
        print("No folds completed successfully.")
        return None, None

    return np.mean(results), best_model

 
def predict_stock(ticker, lookback=60, splits=5, future_steps=1):
    """
    Predicts future stock prices using a trained model and walk-forward validation.
    """
    # Fetch stock data
    stock_data = get_stock_data(ticker)
    if stock_data is None:
        print(f"Failed to retrieve data for ticker: {ticker}")
        return

    # Perform walk-forward validation to train and evaluate the model
    avg_val_loss, best_model = walk_forward_validation(stock_data, lookback, splits)
    if best_model is None:
        print("No model was trained successfully during walk-forward validation.")
        return

    # Prepare the last sequence for prediction
    last_sequence = stock_data.iloc[-lookback:].copy()
    if len(last_sequence) < lookback:
        print(f"Not enough data for prediction. Expected {lookback} rows, got {len(last_sequence)}.")
        return

    print("Last sequence for prediction:")
    print(last_sequence)

    # Preprocess the last sequence
    x_pred, _, scaler = preprocess_data(last_sequence, lookback=lookback)
    if x_pred is None or len(x_pred) == 0:
        print("Error: No valid sequences generated for prediction. Check input data.")
        return

    # Use only the last sequence for prediction and reshape it
    x_pred = np.array([x_pred[-1]])
    
    last_sequence = stock_data.iloc[-lookback:].copy()
    if last_sequence.isnull().values.any():  # Check for NaNs
        print("Error: NaN values found in the last sequence. Cannot predict.")
        print(last_sequence) #Print the sequence to inspect the nan values
        return

    predictions = []
    for _ in range(future_steps):
        # Predict the next price
        predicted_scaled_price = best_model.predict(x_pred)
        predictions.append(predicted_scaled_price[0][0])

        # Create a new sequence for the next prediction
        new_sequence = x_pred[0][1:]  # Remove the oldest timestep
        new_sequence = np.append(new_sequence, predicted_scaled_price[0])  # Append the predicted value
        new_sequence = new_sequence.reshape(1, x_pred.shape[1], x_pred.shape[2])  # Reshape for the model
        x_pred = new_sequence

    # Inverse transform the predictions to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Print the predicted prices
    print(f"Predicted prices for the next {future_steps} days:")
    for i, price in enumerate(predicted_prices):
        print(f"Day {i + 1}: {price[0]}")




# Example usage:
ticker_symbol = input("Enter the ticker symbol: ")
predict_stock(ticker_symbol, future_steps=5)