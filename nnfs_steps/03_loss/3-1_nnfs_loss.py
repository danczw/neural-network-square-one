import numpy as np

# Import basic dataset
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

'''
Multi layer calculation of a Neural Network with batch input
    - added loss function
    - added accuracy
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
    def forward(self, inputs):
        # Get propabilities, minus np.max to prevent overflow problem
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize propabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities

# Define class to initialize loss function
class Loss:
    # Calculate mean loss and pass forward
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define class to initialize categorical cross entropy
class Loss_CategoricalCrossentropy(Loss):
    # Calculate categorical cross entropy
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        '''
        Clip y_pred to prevent inf loss when calculating loss of y_pred = 0
            => see concepts/02_loss.py for details
        '''
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        '''
        Dynamicaly handel confidences for different target var formatting:
            scalar values [1, 0] or one-hot-encoded values[[0, 1], [1, 0]]
        '''
        if len(y_true.shape) == 1: # scalar / categorical values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # one-hot-encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

'''
Initialize layers and activation function
'''
layer_One = Layer_Dense(2, 3)
activation_One = Activation_ReLU()
layer_Two = Layer_Dense(3, 3)
activation_Two = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Pass data through layers, original input is X
layer_One.forward(X)

# Pass output of layer one into activation function
activation_One.forward(layer_One.output)

# Pass output of activation one into layer two
layer_Two.forward(activation_One.output)

# Pass output of layer two into activation function
activation_Two.forward(layer_Two.output)

print(activation_Two.output[:5])

# Calculate loss
loss = loss_function.calculate(activation_Two.output, y)

print(f'loss: {loss}')

# Calculate accuracy from output of layer two activation
predictions = np.argmax(activation_Two.output, axis=1)
# Convert targets if one-hot encoded
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print ('acc:', accuracy)
