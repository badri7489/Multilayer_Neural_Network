import numpy as np
from activation import Activation
from layer import Layer

class Tanh(Activation):
    def __init__(self):
        # making the tanh function
        tanh = lambda x: np.tanh(x)

        # making the derivative of tanh function
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2

        # pass the hyperbolic and its derivative function in the super init
        super().__init__(tanh, tanh_prime)

class Softmax(Layer):
    # Forward Prop for softmax
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    # Backward Prop for softmax
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)