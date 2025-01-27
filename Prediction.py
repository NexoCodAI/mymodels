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

        

