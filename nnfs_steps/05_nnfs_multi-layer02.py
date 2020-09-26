# multi layer calculation of a Neural Network with batch input using OOP

import numpy as np

# changed input naming to X to conform with notation standards
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# define class to initialize layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # keep initital weights close to 0.1 to not create infinitively large number by later propagation through layers
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)        # shape of weights based on input shape and number of neurons
        self.biases = np.zeros((1, n_neurons))                          # shape of biases based on number of neurons, initial biases are set to 0

    def forward(self, inputs):                                          # for 1. layer input is actual input data, every other layer input is self.output of prev layer
        self.output = np.dot(inputs, self.weights) + self.biases        # output is dotproduct + biases calc

'''
initialize layers using layer classes with arguments of shape of input (features) and number of neurons
'''

layer_One = Layer_Dense(4, 5)
# second layer input corresponding with previous number of neurons (and therefore output shape)
layer_Two = Layer_Dense(5, 2)

# pass data through layers
layer_One.forward(X)
print(layer_One.output)

layer_Two.forward(layer_One.output)
print(layer_Two.output)