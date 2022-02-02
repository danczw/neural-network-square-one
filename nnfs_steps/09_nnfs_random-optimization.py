'''
optimization of Neural Network weights and biases
using random guessing
    - semi random guessing weights and biases can lead to better performance
    - >>> slow and only for very basic datasets, i.e. vertical data testset <<<
'''
import matplotlib.pyplot as plt
import numpy as np

import nnfs
nnfs.init()

dataset = 'vertical' # TODO: choose between 'vertical' and 'spiral' dataset

if dataset == 'vertical':
    # import vertical dataset
    from nnfs.datasets import vertical_data
    X, y = vertical_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    # plt.show() # TODO: uncomment to view data
elif dataset == 'spiral':
    # import vertical dataset
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    # plt.show() # TODO: uncomment to view data

# define class to initialize layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # keep initital weights close to 0.1 to not create infinitively large number by later propagation through layers
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)        # shape of weights array based on input shape and number of neurons
        self.biases = np.zeros((1, n_neurons))                          # shape of biases based on number of neurons, initial biases are set to 0

    def forward(self, inputs):                                          # for first layer, input is actual input data (X), every other layer input is self.output of prev layer
        self.output = np.dot(inputs, self.weights) + self.biases        # output is dotproduct + biases calc

# define class to initialize activation function: rectified linear unit
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# define class to initialize activation function: Softmax
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # minus np.max to prevent overflow problem
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# define class to initialize loss function
class Loss: # calculate mean loss and pass forward
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# define class to initialize categorical cross entropy
class Loss_CategoricalCrossentropy(Loss): # calculate categorical cross entropy
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip y_pred to prevent inf loss when calculating loss of y_pred = 0 => see loss.py for details
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # dynamicaly handel confidences for different target var formatting:
        #   scalar values [1, 0] or one-hot-encoded values[[0, 1], [1, 0]]
        if len(y_true.shape) == 1: # scalar
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # one-hot-encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

'''
initialize layers and activation function
'''
layer_One = Layer_Dense(2, 3)
activation_One = Activation_ReLU()
layer_Two = Layer_Dense(3, 3)
activation_Two = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

'''
run n iterations with random generated weights
'''

# initialize loss and accuracy
lowest_loss = 9999999
highest_accuracy = 0

# initialize weights and losses
best_layerOne_weights = layer_One.weights.copy()
best_layerOne_biases = layer_One.biases.copy()
best_layerTwo_weights = layer_Two.weights.copy()
best_layerTwo_biases = layer_Two.biases.copy()

# set iterations for weight and biases guessing
iterations = 100000 # TODO: set iterations

for iteration in range(iterations):
    # semi random guess of weights and biases based on last weights and biases
    layer_One.weights += 0.05 + np.random.rand(2, 3)
    layer_One.biases += 0.05 + np.random.rand(1, 3)
    layer_Two.weights += 0.05 + np.random.rand(3, 3)
    layer_Two.biases += 0.05 + np.random.rand(1, 3)

    # pass data through layer
    layer_One.forward(X)
    activation_One.forward(layer_One.output)
    layer_Two.forward(activation_One.output)
    activation_Two.forward(layer_Two.output)

    # calculate loss
    loss = loss_function.calculate(activation_Two.output, y)

    # calculate accuracy
    predictions = np.argmax(activation_Two.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        # if new model performance (loss) is better (smaller) than previous
        # => copy weights and biases
        best_layerOne_weights = layer_One.weights.copy()
        best_layerOne_biases = layer_One.biases.copy()
        best_layerTwo_weights = layer_Two.weights.copy()
        best_layerTwo_biases = layer_Two.biases.copy()
        lowest_loss = loss # update lowest loss

        highest_accuracy = accuracy # update highest accuracy
        print(highest_accuracy)

    else:
        # else copy previous best weights and biases for next iteration
        layer_One.weights = best_layerOne_weights.copy()
        layer_One.biases = best_layerOne_biases.copy()
        layer_Two.weights = best_layerTwo_weights.copy()
        layer_Two.biases = best_layerTwo_biases.copy()

print(f'---\nfinal accuracy: {round(highest_accuracy * 100, 2)}%')