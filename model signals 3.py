import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image
import cv2 

# Function to load and preprocess PPM images
def load_ppm_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Convert to RGB
    img = img.resize((32, 32))  # Resize to the desired dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Function for background subtraction
def background_subtraction(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    # Find contours and create a mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    if contours:
        # Draw the largest contour
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Bitwise AND to isolate the sign
    isolated_sign = cv2.bitwise_and(image, mask)
    return isolated_sign

# Set the path to the dataset  
data_dir = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\Train'
data_dir1 = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\Train'

# Define parameters
img_height, img_width = 32, 32
batch_size = 5  

# Create an instance of the ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use validation split
)


# Load training data
train_data_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_data_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Check the type of the data generators
print(f'Train data generator type: {type(train_data_gen)}')
print(f'Validation data generator type: {type(val_data_gen)}')

# Check if the validation generator has data
try:
    val_data_sample = next(iter(val_data_gen))
    print(f'Validation data sample shape: {val_data_sample[0].shape}, Labels shape: {val_data_sample[1].shape}')
except StopIteration:
    print("Validation data generator is empty.")

# Proceed to train the model only if data is valid
if train_data_gen.samples > 0 and val_data_gen.samples > 0:
    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), 
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(43, activation='softmax')  # Set this to 62 for your classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=5  # Adjust the number of epochs as needed
    )


# Evaluate the model using the validation generator
try:
    val_loss, val_accuracy = model.evaluate(val_data_gen)
    print(f'Validation accuracy: {val_accuracy:.2f}')
except Exception as e:
    print(f"Error during evaluation: {e}")


# After training the model, get the class labels
class_labels = list(train_data_gen.class_indices.keys())
print(f'Class labels: {class_labels}')



last_trained_model = model

# Save the last trained model in the recommended Keras format
if last_trained_model is not None:
    last_trained_model.save('C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_modelsigns.keras')  # Save as Keras format
