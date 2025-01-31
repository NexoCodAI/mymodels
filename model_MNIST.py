import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split


# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()


# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Normalize the data
def normalize(x):
    x = x.astype('float32') / 255.0  # Scale pixel values to [0, 1]
    return x

# Normalize the datasets
x_train = normalize(x_train)
x_val = normalize(x_val)
x_test = normalize(x_test)

# Reshape the data to include the channel dimension (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
def convert_to_one_hot(labels):
    no_samples = labels.shape[0]
    n_classes = np.max(labels) + 1
    one_hot = np.zeros((no_samples, n_classes))
    one_hot[np.arange(no_samples), labels.ravel()] = 1
    return one_hot

y_train = convert_to_one_hot(y_train)
y_val = convert_to_one_hot(y_val)
y_test = convert_to_one_hot(y_test)


# Define input shape and output size
input_shape = (28, 28, 1)  # Shape of the input images
output_size = 10  # Number of classes in Fashion MNIST

# Example of adding Dropout and Batch Normalization in the model
def build_cnn_with_regularization(input_shape, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model

# Build and compile the updated model
model = build_cnn_with_regularization(input_shape, output_size)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model with data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=60),
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

# Function to check predictions for a few random test images
def check_predictions(model, x_test, y_test, num_images=10):
    # Randomly select indices without a fixed seed
    random_indices = np.random.choice(x_test.shape[0], num_images, replace=False)
    
    # Get the selected images and their true labels
    selected_images = x_test[random_indices]
    selected_labels = y_test[random_indices]
    
    # Make predictions
    predictions = model.predict(selected_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(selected_labels, axis=1)

    print("Predicted classes:", predicted_classes)
    print("True classes:", true_classes)

    # Plot the images with their true and predicted labels
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(selected_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}')
        plt.axis('off')
    plt.show()

last_trained_model = model

# Save the last trained model in the recommended Keras format
if last_trained_model is not None:
    last_trained_model.save('C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_model.keras')  # Save as Keras format
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}')

def plot_predictions(model, x_test, y_test, num_images=10):
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}')
        plt.axis('off')
    plt.show()

# Call the function to plot predictions
plot_predictions(last_trained_model, x_test, y_test)

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your pre-trained model
model = tf.keras.models.load_model('C:\\Users\\jcvis\\OneDrive\\Escritorio\\Ejercicios Phyton\\my_model.keras')

# Load your images
image_path = r'C:\Users\jcvis\Downloads\images (1).jpg'
image_path1 = r'C:\Users\jcvis\Downloads\81P46hSkDsL._AC_UY900_.jpg'

image = Image.open(image_path)
image2 = Image.open(image_path1)

# Resize the images to 28x28
image = image.resize((28, 28))
image2 = image2.resize((28, 28))

# Convert the images to grayscale
image = image.convert('L')
image2 = image2.convert('L')

# Convert the images to numpy arrays
image_array = np.array(image)
image_array2 = np.array(image2)

# Normalize the images
image_array = image_array / 255.0
image_array2 = image_array2 / 255.0

# Reshape the images to (1, 28, 28, 1)
image_array = image_array.reshape(-1, 28, 28, 1)
image_array2 = image_array2.reshape(-1, 28, 28, 1)

# Make predictions
predictions = model.predict(image_array)
predictions2 = model.predict(image_array2)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)
predicted_class2 = np.argmax(predictions2, axis=1)

# Map the predicted class index to the corresponding label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Predicted class index:", predicted_class)
print("Predicted class name:", class_names[predicted_class[0]])
print("Predicted class index:", predicted_class2)
print("Predicted class name:", class_names[predicted_class2[0]])

# Display the processed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {class_names[predicted_class[0]]}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_array2.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {class_names[predicted_class2[0]]}')
plt.axis('off')

plt.show()

        


