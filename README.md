Handwritten Digit Recognition using CNN (Keras)

TASK OVERVIEW -
This task implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset. The model is trained on 28×28 grayscale images and can predict digits from custom images uploaded by the user.

FEATURES - 
Train a CNN model on MNIST digits.
Predict digits from uploaded images.
Save and load trained models (.keras)
Visualize training accuracy and loss.

DATASET -
The project uses the MNIST dataset from tensorflow.keras.datasets, which contains 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28×28 pixels, grayscale.

TECH STACK -

Programming Language: Python3

Libraries: TensorFlow, Keras, NumPy, OpenCV, Matplotlib

Dataset: MNIST Handwritten Digits (tf.keras.datasets.mnist)

Development Environment: Google Colab

IMPLEMENTATION DETAILS -

Data Loading & Preprocessing:
Load MNIST dataset from Keras.
Normalize pixel values.
Reshape images to (28,28,1) for CNN input.
One-hot encode labels for classification.

CNN Architecture:
Conv2D + MaxPooling2D layers to extract features.
Flatten layer to convert 2D features into 1D.
Dense layers with ReLU activation for learning complex patterns.
Dropout layer to prevent overfitting.
Output layer with softmax activation for digit classification (10 classes).

Prediction on New Images:
Users can upload images of handwritten digits.
Images are preprocessed (grayscale, resize to 28×28, normalize).
The trained model predicts the digit and displays the result.
