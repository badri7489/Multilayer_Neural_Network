import numpy as np
from activation import Activation

class Tanh(Activation):
    def __init__(self, activation, activation_prime):
        # making the tanh function
        tanh = lambda x: np.tanh(x)

        # making the derivative of tanh function
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2

        # pass the hyperbolic and its derivative function in the super init
        super.__init__(tanh, tanh_prime)