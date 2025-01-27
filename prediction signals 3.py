import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf

# Function to load and preprocess PPM images
def load_ppm_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Convert to RGB
    img = img.resize((32, 32))  # Resize to the desired dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Load your pre-trained model
model_path = 'C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_modelsigns.keras'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Define class labels (update this list based on your training data)
class_labels =  ['100 kmh', '120 kmh', '20 kmh', '30 kmh', '50 kmh', '60 kmh', '70 kmh', '80 kmh', 'Adelantamiento Prohibido', 
                 'Adelantamiento para camiones prohibido', 'Calzada con prioridad', 'Ceda el Paso', 'Ciclistas', 'Circulación Prohibida', 
                 'Curva Peligrosa Derecha', 'Curva Peligrosa Izquierda', 'Curvas Peligrosas Izquierda', 'Entrada prohibida', 'Estrechamiento por la Derecha', 
                 'Fin de prohibiciones', 'Fin limitacion 80 kmh', 'Fin prohibicion adelantamiento', 'Fin prohibicion adelantamiento para camiones', 
                 'Interseccion con prioridad', 'Niños', 'Obras', 'Otros peligros', 'Paso animales en libertad', 'Paso de peatones', 'Paso obligatorio Derecha', 
                 'Paso obligatorio Izquierda', 'Pavimento Deslizante', 'Pavimento deslizante por hielo o nieve', 'Perfil Irregular', 'Prohibido vehiculos de mercancías', 
                 'Rotonda obligatoria', 'Semáforos', 'Sentido obligatorio derecha', 'Sentido obligatorio izquierda', 'Sentido obligatorio recto', 'Stop', 
                 'Unicas direcciones permitidas R, D', 'Unicas direcciones permitidas R, I'] # Update with actual class names

# Example usage for prediction
def load_and_preprocess_image(img_path):
    img = load_ppm_image(img_path)
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Example usage
img_path = r'C:\Users\jcvis\Downloads\descarga.png'  # Update this path
new_image = load_and_preprocess_image(img_path)

# Make predictions only if the model is loaded successfully
if model is not None:
    predictions = model.predict(new_image)
    predicted_class = np.argmax(predictions)

    # Get the class label
    predicted_label = class_labels[predicted_class]

    # Display the image with the predicted class name
    def display_image_with_label(img_path, label):
        img = Image.open(img_path).convert('RGB')  # Load the original image
        plt.imshow(img)
        plt.title(f'Predicted Class: {label}')
        plt.axis('off')  # Hide axes
        plt.show()

    # Call the function to display the image
    display_image_with_label(img_path, predicted_label)
else:
    print("Model could not be loaded. Prediction cannot be made.")