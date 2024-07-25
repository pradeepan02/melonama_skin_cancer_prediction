Melonama Skin Cancer Prediction
Overview
This project aims to predict melanoma skin cancer using a Convolutional Neural Network (CNN) model. The model is built using the Keras Sequential API and consists of several convolutional layers, max-pooling layers, and dense layers.
Layer Details:
Input Layer: Accepts images of size 128x128 with a single channel (grayscale).
Conv2D Layers: Four convolutional layers with increasing filter sizes (32, 64, 128, 256) and a kernel size of 3x3. Activation function used is "leaky_relu".
MaxPool2D Layers: Four max-pooling layers with a pool size of 2x2 to reduce the spatial dimensions of the feature maps.
Flatten Layer: Flattens the input.
Dense Layers: Two dense layers, one with 256 units and ReLU activation, and the final output layer with a single unit and sigmoid activation for binary classification.
Requirements
To run this project, you need the following libraries installed:

TensorFlow
Keras
NumPy
Matplotlib (optional, for visualizing results)
You can install the required libraries using pip:
pip install tensorflow keras numpy matplotlib
