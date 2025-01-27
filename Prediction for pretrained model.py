import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# Load your trained model (if not already loaded)
model = keras.models.load_model(r'C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_model1.keras')

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    return img_array

# Path to the image you want to predict
img_path = r'C:\Users\jcvis\Downloads\images (1).jpg'  # Replace with your image path
img_array = load_and_preprocess_image(img_path)

# Make predictions
predictions = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)

# Display the image
plt.imshow(image.load_img(img_path))
plt.axis('off')  # Turn off axis
plt.show()

# Print the predicted class
print(f'Predicted class: {predicted_class[0]}')