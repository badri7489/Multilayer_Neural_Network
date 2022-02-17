import time

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# I made these functions
from dense import Dense
from activations import Tanh, Softmax
from losses import mse, mse_prime
from predict import predict, train

# Preprocess the MNIST dataset to be 
def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x / 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)  # one hot encoding done
    y = y.reshape(y.shape[0], 10, 1)

    return x[:limit], y[:limit]

# load MNIST data from the server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# The Neural Network Layers
network = [
    Dense(784, 10),
    Tanh(),
    # Dense(300, 100),
    # Tanh(),
    Dense(10, 10),
    Softmax()
]

# Training time start
start = time.time()

train(network, mse, mse_prime, x_train, y_train, 500, 0.1)

# Training time end
end = time.time()

print('Time taken: ', end - start)

# Predicting the outputs on the Test dataset
ans = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    ans += np.argmax(output) == np.argmax(y)
    # print('Pred: ', np.argmax(output), '\tTrue: ', np.argmax(y))

# Accuracy in Test Dataset
print(ans / 100)