'''
Optimization of Neural Network weights and biases using random guessing
    - added backpropagation
'''
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

import nnfs
nnfs.init()

# TODO: choose between 'vertical' and 'spiral' dataset
dataset = 'spiral'

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
        self.inputs = inputs                                            # remember inputs for creating derivatives during backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases        # output is dotproduct + biases calc

    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)                  # gradient of weights with respect to inputs
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)           # graident of biases with respect to passed-in gradients
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)                  # graident of inputs with respect to weights

# define class to initialize activation function: rectified linear unit
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs                                            # remember inputs for creating derivatives during backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # ReLU() derivative array is filled with:
        #   1s, which do not change the multiplies
        #   and 0s, which zero the multiplying value       
        # => take gradients of subsequent function and set to 0 all that are <=0
        # see concepts/05_backpropagation_single_layer.py for more information
        self.dinputs = dvalues.copy()                                   
        self.dinputs[self.inputs <= 0] = 0                              # Zero gradient where input values were negative

# define class to initialize activation function: Softmax
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs                                            # remember input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # get propabilities, minus np.max to prevent overflow problem
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalize propabilities
        self.output = probabilities
    
    def backward(self, dvalues):
        # create uninitialized array
        self.dinputs = np.empty_like(dvalues)                           # same shape as dvalues

        # enumerate outputs and gradients - see concepts/07_softmax_derivative
        #   for code concept and README.md for mathematical concept
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobin matrix of the output
            jacobin_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobin_matrix, single_dvalue)

# define class to initialize loss function
class Loss:                                                             # calculate mean loss and pass forward
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)                         # calculate sample losses
        data_loss = np.mean(sample_losses)                              # calculate mean loss
        return data_loss

# define class to initialize categorical cross entropy
class Loss_CategoricalCrossentropy(Loss):                               # calculate categorical cross entropy
    def forward(self, y_pred, y_true):
        samples = len(y_pred)                                           # number of samples in batch
        # clip y_pred to prevent inf loss when calculating loss of y_pred = 0
        # => see loss.py for details
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # dynamicaly handel confidences for different target var formatting:
        #   scalar values [1, 0] or one-hot-encoded values[[0, 1], [1, 0]]
        if len(y_true.shape) == 1: # scalar / categorical values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # one-hot-encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)         # calculate losses
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)                                          # number of samples
        labels = len(dvalues[0])                                        # number of labels as per first sample
        # turn numerical labels into one-hot encoded vectors if labels are sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues                                # Calculate gradient
        # optimizers sum all gradients related to each weight and bias before multiplying them
        # more samples => more gradients => bigger sum => adjustment of learning rate needed
        # solution: normalization of values by calculating their mean
        self.dinputs = self.dinputs / samples                           # Normalize gradient 

'''
Common Categorical Cross-Entropy loss and Softmax activation
- previously, Categorical Cross-Entropy loss function partial derivative and
    Softmax activation function partial derivative where implemented separately
- for simpler and faster execution, a wholistic implementation can be used based
    again on the chain rule
- called common Categorical Cross-entropy loss and Softmax activation derivative
- see README.md for mathematical concept
'''
# Softmax classifier - combined Softmax activation and cross-entropy loss function
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    # Create activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)                                          # number of samples
        # turn one-hot encoded vectors into numerical labels
        if len(y_true.shape) == "":
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()                                   # copy to safely modify
        self.dinputs[range(samples), y_true] -= 1                       # calculate gradient
        self.dinputs = self.dinputs / samples                           # normalize gradients        

'''
test if combined backward step returns same values
compared to seperate backpropagetion
'''
# create dummy data
softmax_outputs = np.array([[0.7, 0.1, 0.2],                            # 3 probabilities for each class for 3 samples
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])                                     # true classes

#separate backward step
def backward_separate():
    activation = Activation_Softmax()                                   # initialize softmax function
    activation.output = softmax_outputs                                 # set softmax outputs
    loss = Loss_CategoricalCrossentropy()                               # initialize loss function
    loss.backward(softmax_outputs, class_targets)                       # backpropagete outputs based on true class through loss function
    activation.backward(loss.dinputs)                                   # backpropagate loss gradients through activation function
    return activation.dinputs

# combined backward step
def backward_combined():
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()    # initialize combined function 
    softmax_loss.backward(softmax_outputs, class_targets)               # backpropagete outputs based on true class
    return softmax_loss.dinputs

# get results and time execution
t_combined = timeit(lambda: backward_combined(), number = 10000)
t_separate = timeit(lambda: backward_separate(), number = 10000)

print(f'Gradients: separate loss and activation (runtime: {round(t_separate, 4)}):\n', backward_separate())
print(f'Gradients: combined loss and activation (runtime: {round(t_combined, 4)}):\n', backward_combined())
print(f'Separate gradient calculation was about {round(t_separate/t_combined, 2)} slower than combined calculation\n')

'''
initialize layers and activation function using
common Categorical Cross-Entropy loss and Softmax activation
'''
layer_One = Layer_Dense(2, 3)
activation_One = Activation_ReLU()
layer_Two = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# forward pass data through layers
layer_One.forward(X)                                                    # original input is X
activation_One.forward(layer_One.output)                                # pass output of layer one into activation function
layer_Two.forward(activation_One.output)
loss = loss_activation.forward(layer_Two.output, y)                     # pass output of layer two through activation and loss function and calculate loss

print(loss_activation.output[:5], ' (propapilities of first 5 samples)') # output of first few samples
print('loss: ', loss)

predictions = np.argmax(loss_activation.output, axis=1)                 # get predictions by finding index of highest class confidence
if len(y.shape) == 2:                                                   # convert targets if one-hot encoded
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)                                      # calculate accuracy

print('accuracy: ', accuracy, '\n')

# backward pass data through layers
loss_activation.backward(loss_activation.output, y)                     # backpropagete outputs based on true class through loss and softmax function
layer_Two.backward(loss_activation.dinputs)                             # backpropagate softmax function derivatives through layer two
activation_One.backward(layer_Two.dinputs)                              # backpropagate layer two derivatives through activation one (relu)
layer_One.backward(activation_One.dinputs)                              # backpropagate activation one (relu) derivatives through layer one

print(layer_One.dweights, ' (gradients of layer one weights)')
print(layer_One.dbiases, ' (gradients of layer one biases)')
print(layer_Two.dweights, ' (gradients of layer two weights)')
print(layer_One.dbiases, ' (gradients of layer two biases)')
