import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam  # Import the Adam optimizer

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to (28, 28, 1) and then resize to (42, 42, 3) for MobileNetV2
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = np.concatenate([x_train, x_train, x_train], axis=-1)  # Convert to 3 channels
x_val = np.concatenate([x_val, x_val, x_val], axis=-1)
x_test = np.concatenate([x_test, x_test, x_test], axis=-1)

x_train = tf.image.resize(x_train, (42, 42))
x_val = tf.image.resize(x_val, (42, 42))
x_test = tf.image.resize(x_test, (42, 42))

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_val = keras.utils.to_categorical(y_val, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Load the MobileNetV2 model without the top layers
base_model = keras.applications.MobileNetV2(input_shape=(42, 42, 3), include_top=False, weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create the model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes for Fashion MNIST
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Specify optimizer here
              loss='categorical_crossentropy',  # Use categorical_crossentropy
              metrics=['accuracy'])  # Specify metrics to track

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),  # Pass the correct data format
    epochs=5,
    validation_data=(x_val, y_val)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}')

# Save the model after initial training
model.save(r'C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_model1_initial.keras')

# Optional: Fine-tuning the model
# Unfreeze some layers of the base model
base_model.trainable = True

# Compile the model again with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),  # Use a smaller learning rate for fine-tuning
              loss='categorical_crossentropy',  # Use categorical_crossentropy
              metrics=['accuracy'])  # Specify metrics to track

