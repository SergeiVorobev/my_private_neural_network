import os
import numpy as np
import tensorflow as tf
import sys

# Get the class name from the command line argument
if len(sys.argv) != 2:
    print("Usage: python test_model.py <class_name>")
    print("Class names: plus, minus, dot, division")
    sys.exit(1)

class_name = sys.argv[1]
class_mapping = {
    'minus': 'minus.png',
    'plus': 'plus.png',
    'dot': 'dot.png',
    'division': 'division.png'
}

# Validate the class name
if class_name not in class_mapping:
    print(f"Invalid class name: {class_name}. Please choose from: {', '.join(class_mapping.keys())}.")
    sys.exit(1)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'my_model_v1_0.h5')
image_dir = os.path.join(base_dir, '..', 'img')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load and preprocess the test image
test_image_path = os.path.join(image_dir, class_mapping[class_name])
test_image = tf.keras.utils.load_img(test_image_path, color_mode='grayscale', target_size=(3, 3))
test_image_array = tf.keras.utils.img_to_array(test_image)
test_image_array = test_image_array.flatten() / 255.0  # Normalize and flatten
test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension

# Predict the class of the new image
prediction = model.predict(test_image_array)
predicted_class_index = np.argmax(prediction)
predicted_class_probability = prediction[0][predicted_class_index]

# Class mapping for output
output_mapping = {0: 'minus', 1: 'plus', 2: 'dot', 3: 'division'}

# Get the predicted symbol
predicted_symbol = output_mapping[predicted_class_index]

# Print the results
print(f"Predicted symbol: {predicted_symbol}")
print(f"Predicted probabilities: {prediction}")
print(f"Prediction accuracy: {predicted_class_probability:.2%}")
