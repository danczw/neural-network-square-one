import matplotlib.pyplot as plt
import numpy as np

# Import basic dataset
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

'''
Multi layer calculation of a Neural Network with batch input
    - added foreward pass and activation function for first layer
'''

# Define input
X, y = spiral_data(100, 3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Define class to initialize layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Shape of weights array based on input shape and number of neurons
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        
        # Shape of biases based on number of neurons, initial biases are set to 0
        self.biases = np.zeros((1, n_neurons))

    # For first layer, input is actual input data (X), every other layer input is self.output of prev layer
    def forward(self, inputs):
        # Output is dotproduct + biases calc
        self.output = np.dot(inputs, self.weights) + self.biases

# Define class to initialize activation function: rectified linear unit
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''
Initialize layers and activation function
    - for simplicity, only layer one is focused at this point
'''
layer_One = Layer_Dense(2, 5)
activation_One = Activation_ReLU()

# Pass data through layers, original input is X
layer_One.forward(X)

# Pass output of layer one into activation function
activation_One.forward(layer_One.output)

print(activation_One.output)
