import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt 
from tqdm import tqdm

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
(60000,28,28)
(60000,)
(10000,28,28)
(10000,)

x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)

def normalize(x_train, x_test):
  train_mean = np.mean(x_train)
  train_std = np.mean(x_train)
  x_train = (x_train - train_mean)/train_std
  x_test = (x_test - train_mean)/train_std  
  return x_train, x_test

def convert_to_one_hot(labels):
  no_samples = labels.shape[0]
  n_classes = np.max(labels) + 1
  one_hot = np.zeros((no_samples, n_classes))
  one_hot[np.arange(no_samples),labels.ravel()] = 1
  return one_hot

x_train, x_test = normalize(x_train, x_test)
y_train = convert_to_one_hot(y_train)
y_test = convert_to_one_hot(y_test)

def get_placeholders(input_size, output_size):
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="inputs")
  targets = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name="targets")
  return inputs, targets

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def dense_layer(inputs, hidden_units, activation_fn):
    # Create a Dense layer
    layer = tf.keras.layers.Dense(units=hidden_units, activation=activation_fn)(inputs)
    return layer

def build_network(input_size, output_size, hidden_units, num_layers, activation_fn):
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs
    
    # Add hidden layers
    for units in hidden_units:
        x = dense_layer(x, units, activation_fn)
    
    # Output layer
    outputs = tf.keras.layers.Dense(units=output_size, activation='softmax')(x)  # Change activation as needed
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(features, labels, hidden_units, epochs, batch_size, learning_rate, num_layers, activation_fn):
    input_size = features.shape[1]
    output_size = labels.shape[1]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
    model = build_network(input_size, output_size, hidden_units, num_layers, activation_fn)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()  # or any other loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        total_accuracy = 0
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            total_accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_batch, axis=1)), tf.float32)).numpy()

        losses.append(epoch_loss / len(dataset))
        accuracies.append(total_accuracy / len(dataset))
        print("Epoch: {}/{} , Loss: {} , Accuracy: {}".format(epoch + 1, epochs, losses[-1], accuracies[-1]))

    return losses, accuracies

# Define your features and labels
features = x_train
labels = y_train
epochs = 25
batch_size = 256 
learning_rate = 0.001
num_layers = 9
hidden_units = [30, 30, 30, 30, 30]
input_units = x_train.shape[1]
output_units = y_train.shape[1] 

# Run the model for the activation functions sigmoid and relu
activation_fns = {"swish": 'swish', "sigmoid": 'sigmoid', "relu": 'relu',}
loss = {}
acc = {}

for name, activation_fn in activation_fns.items():
    model_name = "Running model with activation function as {}".format(name)
    print(model_name)
    print(features, labels)
    losses, accuracies = train_model(features=x_train, labels=y_train, hidden_units=hidden_units, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_layers=num_layers, activation_fn=activation_fn)
    acc[name] = accuracies
    
     
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