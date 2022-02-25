'''
Optimization of Neural Network weights and biases using random guessing
    - added backpropagation
'''
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

# Import basic dataset
import nnfs
nnfs.init()
# TODO: choose between 'vertical' and 'spiral' dataset
dataset = 'vertical'

if dataset == 'vertical':
    # Import vertical dataset
    from nnfs.datasets import vertical_data
    X, y = vertical_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    plt.show()
elif dataset == 'spiral':
    # Import vertical dataset
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    plt.show()

# Define class to initialize layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        Keep initital weights close to 0.1 to not create
            infinitively large number by later propagation through layers
        '''
        # Shape of weights array based on input shape and number of neurons
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        
        # Shape of biases based on number of neurons, initial biases are set to 0
        self.biases = np.zeros((1, n_neurons))

    # For first layer, input is actual input data (X), every other layer input is self.output of prev layer
    def forward(self, inputs):
        # Remember inputs for creating derivatives during backpropagation
        self.inputs = inputs
        # Output is dotproduct + biases calc
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient of weights with respect to inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # Graident of biases with respect to passed-in gradients
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Graident of inputs with respect to weights
        self.dinputs = np.dot(dvalues, self.weights.T)

# Define class to initialize activation function: rectified linear unit
class Activation_ReLU:
    def forward(self, inputs):
        # Remember inputs for creating derivatives during backpropagation
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''
        ReLU() derivative array is filled with:
            - 1s, which do not change the multiplies
            - and 0s, which zero the multiplying value       
            => take gradients of subsequent function and set to 0 all that are <=0
            - see concepts/05_backpropagation_single_layer.py for more information
        '''
        self.dinputs = dvalues.copy()     

        # Zero gradient where input values were negative                              
        self.dinputs[self.inputs <= 0] = 0

# Define class to initialize activation function: Softmax
class Activation_Softmax:
    '''
    Use of Softmax to exponentiate and normalize values to get
        interpretable output, i.e. probability between 0 and 1
    '''
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get propabilities, minus np.max to prevent overflow problem
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize propabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
    
    def backward(self, dvalues):
        # Create uninitialized array with same shape as dvalues
        self.dinputs = np.empty_like(dvalues)

        '''
        Enumerate outputs and gradients => see concepts/07_softmax_derivative.py
            for code concept and README.md for mathematical concept
        '''
        for index, (single_output, single_dvalue) in enumerate(zip(self.output,
                                                                   dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobin matrix of the output
            jacobin_matrix = np.diagflat(single_output) - np.dot(single_output,
                                                                 single_output.T)
            
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobin_matrix, single_dvalue)

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

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of labels as per first sample
        labels = len(dvalues[0])
        
        # Turn numerical labels into one-hot encoded vectors if labels are sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        '''
        Optimizers sum all gradients related to each weight and bias
            before multiplying them
        More samples => more gradients => bigger sum => adjustment of
            learning rate needed
        Solution: normalization of values by calculating their mean
        '''
        self.dinputs = self.dinputs / samples

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
        # Number of samples
        samples = len(dvalues)

        # Turn one-hot encoded vectors into numerical labels
        if len(y_true.shape) == "":
            y_true = np.argmax(y_true, axis=1)
        
        # Copy to safely modify
        self.dinputs = dvalues.copy()

        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize gradients
        self.dinputs = self.dinputs / samples

'''
Test if combined backward step returns same values
    compared to seperate backpropagetion
'''
 # 3 probabilities for each class for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
# True classes
class_targets = np.array([0, 1, 1])

# Separate backward step
def backward_separate():
    # Initialize softmax function
    activation = Activation_Softmax()

    # Set softmax outputs
    activation.output = softmax_outputs

    # Initialize loss function
    loss = Loss_CategoricalCrossentropy()

    # Backpropagete outputs based on true class through loss function
    loss.backward(softmax_outputs, class_targets)

    # Backpropagate loss gradients through activation function
    activation.backward(loss.dinputs)

    return activation.dinputs

# Combined backward step
def backward_combined():
    # Initialize combined function 
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()

    # Backpropagete outputs based on true class
    softmax_loss.backward(softmax_outputs, class_targets)

    return softmax_loss.dinputs

# Get results and time execution
t_combined = timeit(lambda: backward_combined(), number = 10000)
t_separate = timeit(lambda: backward_separate(), number = 10000)

print(f'Gradients: separate loss and activation (runtime: {round(t_separate, 4)}):\n', backward_separate())
print(f'Gradients: combined loss and activation (runtime: {round(t_combined, 4)}):\n', backward_combined())
print(f'Separate gradient calculation was about {round(t_separate/t_combined, 2)} slower than combined calculation\n')

'''
Initialize layers and activation function using
    common Categorical Cross-Entropy loss and Softmax activation
'''
layer_One = Layer_Dense(2, 3)
activation_One = Activation_ReLU()
layer_Two = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# Pass data through layers, original input is X
layer_One.forward(X)

# Pass output of layer one into activation function
activation_One.forward(layer_One.output)

# Pass output of activation one into layer two
layer_Two.forward(activation_One.output)

# Pass output of layer two into combined loss activation function
loss = loss_activation.forward(layer_Two.output, y)

# Output of first few samples
print(loss_activation.output[:5], ' (propapilities of first 5 samples)')
print('loss: ', loss)

# Get predictions by finding index of highest class confidence
predictions = np.argmax(loss_activation.output, axis=1)

# Convert targets if one-hot encoded
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('accuracy: ', accuracy, '\n')

# Backpropagete outputs based on true class through loss and softmax function
loss_activation.backward(loss_activation.output, y)

# Backpropagate softmax function derivatives through layer two
layer_Two.backward(loss_activation.dinputs)

# Backpropagate layer two derivatives through activation one (relu)
activation_One.backward(layer_Two.dinputs)

# Backpropagate activation one (relu) derivatives through layer one
layer_One.backward(activation_One.dinputs)

print(layer_One.dweights, ' (gradients of layer one weights)')
print(layer_One.dbiases, ' (gradients of layer one biases)')
print(layer_Two.dweights, ' (gradients of layer two weights)')
print(layer_One.dbiases, ' (gradients of layer two biases)')
