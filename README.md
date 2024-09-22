# Neural Network for Symbol Recognition

## Overview

This project implements a simple neural network to recognize mathematical symbols (*minus*, *plus*, *dot*(it represents multiplication), and *division*) from images. The model processes images stored in a designated folder and classifies them into one of four categories.

## Features

- **Input Layer:** Accepts 3x3 pixel grayscale images.
- **Hidden Layer:** Contains 5 neurons with ReLU activation.
- **Output Layer:** Classifies images into four categories using softmax activation.
- **Model Training:** The model is trained on a dataset of labeled images and can be evaluated for accuracy.

## Current Status

As of now, the model achieves approximately 30% accuracy in predictions. Future improvements will focus on increasing the number of neurons, tuning hyperparameters, and potentially adding more layers to enhance performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SergeiVorobev/my_private_neural_network.git
2. Navigate to the project directory:
   ```bash
   cd my_private_neural_network
3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Generate the images using the `create_images.py` script:
   ```bash
   python scripts/create_images.py
2. Train the model by running the main.py script:
   ```bash
   python scripts/main.py
3. Test the model with specific symbols using the test_model.py script:
   ```bash
   python scripts/test_model.py <symbol>
4. Replace <symbol> with one of the following: plus, minus, dot, or division.

## Contribution

Contributions are welcome! If you have ideas for improvements or enhancements, please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
