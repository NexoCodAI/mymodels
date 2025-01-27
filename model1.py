import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape the data
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

# Normalize the data
def normalize(x_train, x_test):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)  # Use std instead of mean for normalization
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std  
    return x_train, x_test

import numpy as np
import matplotlib.pyplot as plt

# Assuming y_train contains your training labels
plt.hist(y_train, bins=np.arange(11) - 0.5, rwidth=0.8)  # 10 classes + 1 for the right edge
plt.xticks(range(10), ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Class Distribution')
plt.show()

# Convert labels to one-hot encoding
def convert_to_one_hot(labels):
    no_samples = labels.shape[0]
    n_classes = np.max(labels) + 1
    one_hot = np.zeros((no_samples, n_classes))
    one_hot[np.arange(no_samples), labels.ravel()] = 1
    return one_hot

x_train, x_test = normalize(x_train, x_test)
y_train = convert_to_one_hot(y_train)
y_test = convert_to_one_hot(y_test)

# Build the neural network
def dense_layer(inputs, hidden_units, activation_fn):
    layer = tf.keras.layers.Dense(units=hidden_units, activation=activation_fn)(inputs)
    return layer

def build_network(input_size, output_size, hidden_units, num_layers, activation_fn):
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs
    
    # Add hidden layers
    for units in hidden_units:
        x = dense_layer(x, units, activation_fn)
    
    # Output layer
    outputs = tf.keras.layers.Dense(units=output_size, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Train the model
def train_model(features, labels, hidden_units, epochs, batch_size, learning_rate, num_layers, activation_fn):
    input_size = features.shape[1]
    output_size = labels.shape[1]

    model = build_network(input_size, output_size, hidden_units, num_layers, activation_fn)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Fit the model
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history.history['loss'], history.history['accuracy']  # Return the model and history

# Define your features and labels
features = x_train
labels = y_train
epochs = 50
batch_size = 3000
learning_rate = 0.001
num_layers = 3
hidden_units = [15, 15, 15]

# Run the model for different activation functions
activation_fns = {"swish": 'swish', "sigmoid": 'sigmoid', "relu": 'relu'}
acc = {}

# Initialize a variable to hold the last trained model
last_trained_model = None

for name, activation_fn in activation_fns.items():
    model_name = "Running model with activation function as {}".format(name)
    print(model_name)
    model, losses, accuracies = train_model(features=x_train, labels=y_train, hidden_units=hidden_units, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_layers=num_layers, activation_fn=activation_fn)
    acc[name] = accuracies
    last_trained_model = model  # Store the last trained model
    
def plot_accuracy(accuracies, title):
    print("Accuracies Dictionary:", accuracies)  # Debugging line
    for name, values in accuracies.items():
        if values:  # Check if there are values to plot
            plt.plot(values, label=name)  # Ensure each line has a label
        else:
            print(f"No values to plot for {name}")  # Debugging line
    plt.legend(title=title)  # This will now work correctly
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

# Call the plot function after training
plot_accuracy(acc, "Model Accuracy")

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

# Evaluate the model on a test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')
# Check predictions for a few test images
def check_predictions(model, x_test, y_test, num_images=10):
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("Predicted classes:", predicted_classes[:num_images])
    print("True classes:", true_classes[:num_images])

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}')
        plt.axis('off')
    plt.show()

# Call the function to check predictions
check_predictions(last_trained_model, x_test, y_test)

