import numpy as np
from layer import Layer

class Activation(Layer):

    # Initialising Weights and Biases
    def __init__(self, activation, activation_prime):
        # initialising activation function
        self.activation = activation

        # initialising derivative of activation function
        self.activation_prime = activation_prime

    # Forward Pass
    def forward(self, input):
        # initialising input
        self.input = input

        # Applying activation function and returning output
        return self.activation(self.input)

    # Backward Pass
    def backward(self, output_gradient, learning_rate):
        # Gradient for the next layer dE/dX
        return np.multiply(output_gradient, self.activation_prime(self.input))