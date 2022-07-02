import numpy as np

# Import basic dataset
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

'''
Multi layer calculation of a Neural Network with batch input
    - added activation function for output layer
'''

# Define input
X, y = spiral_data(samples=100, classes=3)

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

# Define class to initialize activation function: Softmax
class Activation_Softmax:
    '''
    Use of Softmax to exponentiate and normalize values to get
        interpretable output, i.e. probability between 0 and 1
    '''
    def forward(self, inputs):
        # Get propabilities, minus np.max to prevent overflow problem
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize propabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities

'''
Initialize layers and activation function
'''
layer_One = Layer_Dense(2, 3)
activation_One = Activation_ReLU()
layer_Two = Layer_Dense(3, 3)
activation_Two = Activation_Softmax()

# Pass data through layers, original input is X
layer_One.forward(X)

# Pass output of layer one into activation function
activation_One.forward(layer_One.output)

# Pass output of activation one into layer two
layer_Two.forward(activation_One.output)

# Pass output of layer two into activation function
activation_Two.forward(layer_Two.output)

print(activation_Two.output[:5])
