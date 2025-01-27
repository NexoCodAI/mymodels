import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Load the CSV file into a DataFrame
data = pd.read_csv(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Check for missing values and drop them
data.dropna(inplace=True)

# Select only numeric columns for features
X = data.select_dtypes(include=[np.number])

# Drop the 'Year' column and the target column from the features
X = X.drop(columns=['Year', 'Life_expectancy'], errors='ignore')
y = data['Life_expectancy'].values  # Target (life expectancy)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance and scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\scaler.pkl')

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save the PCA model
joblib.dump(pca, r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\pca_model.pkl')

# Define a function to create the model with hyperparameters
def create_model(activation='relu', neurons=128, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(X_train_pca.shape[1],)),
        layers.Dense(neurons, activation=activation),
        layers.Dropout(0.5),
        layers.Dense(neurons, activation=activation, kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Custom Estimator Class
class KerasEstimator:
    def __init__(self, activation='relu', neurons=128, learning_rate=0.001, batch_size=32, epochs=100):
        self.activation = activation
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
    
    def fit(self, X, y):
        # Create and compile the model
        self.model = create_model(activation=self.activation, 
                                  neurons=self.neurons, 
                                  learning_rate=self.learning_rate)
        
        # Train the model
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        
    def predict(self, X):
        # Predict using the trained model
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        # Return a dictionary of the hyperparameters
        return {
            'activation': self.activation,
            'neurons': self.neurons,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
    
    def set_params(self, **params):
        # Set the hyperparameters from a dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Define the parameter grid for Random Search
param_dist = {
    'activation': ['relu', 'tanh', 'swish'],
    'neurons': [32, 64, 128, 256, 512],
    'batch_size': [8, 16, 32, 64],
    'epochs': [100 , 250, 300, 400],
    'learning_rate': [0.0001, 0.001, 0.01]
}

# Wrap the Keras model with the custom estimator
model = KerasEstimator()

# Perform RandomizedSearchCV directly using the custom estimator
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10, cv=3,
    n_jobs=-1, scoring='neg_mean_squared_error'
)

# Fit the model
random_search.fit(X_train_pca, y_train)

# Print the best parameters from Random Search
print("Best parameters from Random Search:", random_search.best_params_)

# Evaluate the best model
best_model = KerasEstimator(**random_search.best_params_)
best_model.fit(X_train_pca, y_train)

# Get predictions
predictions = best_model.predict(X_test_pca)

# Calculate R2 and MSE
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"R2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Plotting Mean Test Score vs. Number of Neurons
results = random_search.cv_results_
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(results_df['param_neurons'], results_df['mean_test_score'], alpha=0.5)
plt.xlabel('Number of Neurons')
plt.ylabel('Mean Test Score')
plt.title('Mean Test Score vs. Number of Neurons')

plt.subplot(1, 2, 2)
plt.scatter(results_df['param_batch_size'], results_df['mean_fit_time'], alpha=0.5)
plt.xlabel('Batch Size')
plt.ylabel('Mean Fit Time (seconds)')
plt.title('Mean Fit Time vs. Batch Size')

plt.tight_layout()
plt.show()

# Plot Actual vs Predicted Life Expectancy
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()

# Save the best model
best_model.model.save(r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\best_life_expectancy_model.keras')
