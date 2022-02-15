import numpy as np
from layer import Layer

class Dense(Layer):

    # Initialising Weights and Biases
    def __init__(self, input_size, output_size):
        # Weight vector of size j x i
        self.weights = np.random.randn(output_size, input_size)
        
        # Bias vector of size j x 1
        self.bias = np.random.randn(output_size, 1)

    # Forward Pass
    def forward(self, input):
        # initialising input
        self.input = input

        # Output
        return np.dot(self.weights, self.input) + self.bias

    # Backward Pass
    def backward(self, output_gradient, learning_rate):
        # dE/dW
        weight_gradient = np.dot(output_gradient, self.input.T)

        # dE/dB
        bias_gradient = output_gradient

        # Updating weights
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

        # Gradient for the next layer dE/dX
        return np.dot(self.weights.T, output_gradient)