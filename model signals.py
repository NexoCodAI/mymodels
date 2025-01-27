import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image

# Function to load and preprocess PPM images
def load_ppm_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Convert to RGB
    img = img.resize((32, 32))  # Resize to the desired dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Set the path to the dataset
data_dir = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\BelgiumTSC_Training\Training'
data_dir1 = r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\BelgiumTSC_Testing\Testing'

# Define parameters
img_height, img_width = 32, 32
batch_size = 32

# Create an instance of the ImageDataGenerator
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
        layers.Dense(62, activation='softmax')  # Set this to 62 for your classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=10  # Adjust the number of epochs as needed
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


import matplotlib.pyplot as plt

# Example usage for prediction
def load_and_preprocess_ppm_image(img_path):
    img = load_ppm_image(img_path)
    return np.expand_dims(img, axis=0)

# Example usage
img_path = r'C:\Users\jcvis\Downloads\senal-peligro-pavimento-deslizante.jpg'  # Update this path
new_image = load_and_preprocess_ppm_image(img_path)
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions)

# Get the class label
class_labels = list(train_data_gen.class_indices.keys())
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
