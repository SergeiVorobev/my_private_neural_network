import os
import numpy as np
from keras import layers
from keras import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 1. Loading images from the directory

# Define the directory containing images (automatically loading from the project structure)
image_dir = 'img'

# Define image dimensions and the number of classes
img_width, img_height = 3, 3
num_classes = 4

# Initialize lists to hold image data and labels
X = []
y = []

# Define class labels (matching subdirectories in the images folder)
class_labels = ['minus', 'plus', 'dot', 'division']

# Loop through each class and load corresponding images
for label_index, label in enumerate(class_labels):
    label_dir = os.path.join(image_dir, label)
    if not os.path.exists(label_dir):
        print(f"Directory for class '{label}' not found in the specified path.")
        continue

    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            # Load and preprocess image (grayscale, resize to 3x3)
            img = load_img(img_path, color_mode='grayscale', target_size=(img_height, img_width))
            img_array = img_to_array(img)
            img_array = img_array.flatten() / 255.0  # Normalize and flatten

            X.append(img_array)
            y.append(label_index)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Define the model using the Sequential API as in the official example

model = Sequential()

# Add input layer (input shape is 9, because the flattened 3x3 image results in 9 features)
model.add(layers.Input(shape=(9,)))

# Add a hidden layer with 5 neurons and ReLU activation
model.add(layers.Dense(5, activation="relu", name="hidden_layer_1"))

# Add the output layer with 4 neurons (for the 4 classes: minus, plus, dot, division)
model.add(layers.Dense(num_classes, activation="softmax", name="output_layer"))

# Print model summary
model.summary()

# 3. Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# 5. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 6. Test the model with a new example

# Assume you have a new test image (for example, a new "plus" image)
test_image_path = 'project/images/plus/example_image.png'

# Load and preprocess the test image
test_image = load_img(test_image_path, color_mode='grayscale', target_size=(img_height, img_width))
test_image_array = img_to_array(test_image)
test_image_array = test_image_array.flatten() / 255.0  # Normalize and flatten
test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension

# Predict the class of the new image
prediction = model.predict(test_image_array)
print("Predicted probabilities:", prediction)
print("Predicted class:", np.argmax(prediction))  # Returns the class with the highest probability
