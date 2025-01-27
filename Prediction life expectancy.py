import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import joblib

# Load your pre-trained model
model_path = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\Life expectancy.keras'
try:
    best_model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    best_model = None  # Set model to None if loading fails

# Load the scaler for making predictions later
try:
    loaded_scaler = joblib.load(r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\scaler.pkl')
except Exception as e:
    print(f"Error loading scaler: {e}")
    loaded_scaler = None  # Set scaler to None if loading fails

# Define the input data for a new country (example values)
input_data = {
    'Infant_deaths': 500, 
    'Under_five_deaths': 400,             
    'Adult_mortality': 600,             
    'Alcohol_consumption': 10,         
    'Hepatitis_B': 100,       
    'Measles': 80,       
    'BMI': 80,      
    'Polio': 90,
    'Diphtheria': 90,    
    'Incidents_HIV': 0.05,
    'GDP_per_capita': 100, 
    'Population_mln': 30,  
    'Thinness_ten_nineteen_years': 10, 
    'Thinness_five_nine_years': 6,
    'Schooling': 6,
    'Economy_status': 0,  # Ensure this feature is valid
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])  # This is correct

# Scale the input data using the loaded scaler
if loaded_scaler is not None:
    input_scaled = loaded_scaler.transform(input_df)

    # If you applied PCA during training, you need to load the PCA model and transform the input
    pca_model_path = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\pca_model.pkl'  # Adjust the path as needed
    try:
        pca_model = joblib.load(pca_model_path)
        input_pca = pca_model.transform(input_scaled)  # Transform the scaled input data
    except Exception as e:
        print(f"Error loading PCA model: {e}")
        input_pca = input_scaled  # Fallback to scaled input if PCA loading fails

    # Make predictions using the trained model
    predicted_life_expectancy = best_model.predict(input_pca)

    # Print the predicted life expectancy
    print(f'Predicted Life Expectancy: {predicted_life_expectancy[0][0]}')
else:
    print("Scaler not loaded. Cannot scale input data.")