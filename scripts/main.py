import os
import numpy as np
import tensorflow as tf

# Define the directory containing images
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, '..', 'img')

# Define image dimensions and the number of classes
img_width, img_height = 3, 3
num_classes = 4

# Initialize lists to hold image data and labels
X = []
y = []

# Mapping file names to class labels
class_mapping = {
    'minus.png': 0,
    'plus.png': 1,
    'dot.png': 2,  # dot represents multiplication
    'division.png': 3
}

# Loop through each image file in the directory and load it
for img_file in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_file)

    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
        # Load and preprocess image (grayscale, resize to 3x3)
        img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array.flatten() / 255.0  # Normalize and flatten

        X.append(img_array)

        # Assign the correct label based on the file name
        if img_file in class_mapping:
            y.append(class_mapping[img_file])
        else:
            print(f"Unrecognized file: {img_file}, skipping.")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split the data into training and test sets
train_size = int(0.75 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the updated model with additional hidden layers and more neurons
model = tf.keras.Sequential()

# Input layer (input shape is 9, because the flattened 3x3 image results in 9 features)
model.add(tf.keras.layers.Input(shape=(9,)))

# Hidden layer 1 with 64 neurons
model.add(tf.keras.layers.Dense(64, activation="relu", name="hidden_layer_1"))

# Hidden layer 2 with 128 neurons
model.add(tf.keras.layers.Dense(128, activation="relu", name="hidden_layer_2"))

# Hidden layer 3 with 64 neurons
model.add(tf.keras.layers.Dense(64, activation="relu", name="hidden_layer_3"))

# Output layer with 4 neurons (for the 4 classes: minus, plus, dot, division)
model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer"))

# Print model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Save the model
model_dir = os.path.join(base_dir, '..', 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, 'my_model_v2_0.h5'))  # Save the trained model
