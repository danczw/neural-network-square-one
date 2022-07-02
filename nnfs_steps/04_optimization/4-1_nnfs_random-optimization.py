import matplotlib.pyplot as plt
import numpy as np

# Import basic dataset
import nnfs
nnfs.init()

'''
Optimization of Neural Network weights and biases using random guessing
    - semi random guessing weights and biases can lead to better performance
    - >>> slow and only for very basic datasets, i.e. vertical data testset <<<
'''

# TODO: choose between 'vertical' and 'spiral' dataset
dataset = 'vertical'

if dataset == 'vertical':
    # Import vertical dataset
    from nnfs.datasets import vertical_data
    X, y = vertical_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    plt.show()
elif dataset == 'spiral':
    # Import spiral dataset
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
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
        
        # Clip y_pred to prevent inf loss when calculating loss of y_pred = 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Dynamicaly handel confidences for different target var formatting
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

'''
Run n iterations with random generated weights
'''

# Initialize loss and accuracy
lowest_loss = 9999999
highest_accuracy = 0

# Initialize weights and losses
best_layerOne_weights = layer_One.weights.copy()
best_layerOne_biases = layer_One.biases.copy()
best_layerTwo_weights = layer_Two.weights.copy()
best_layerTwo_biases = layer_Two.biases.copy()

# Set iterations for weight and biases guessing, TODO: set iterations
iterations = 100000

for iteration in range(iterations):
    # Semi random guess of weights and biases based on last weights and biases
    layer_One.weights += 0.05 + np.random.rand(2, 3)
    layer_One.biases += 0.05 + np.random.rand(1, 3)
    layer_Two.weights += 0.05 + np.random.rand(3, 3)
    layer_Two.biases += 0.05 + np.random.rand(1, 3)

    # Pass data through layers, original input is X
    layer_One.forward(X)

    # Pass output of layer one into activation function
    activation_One.forward(layer_One.output)

    # Pass output of activation one into layer two
    layer_Two.forward(activation_One.output)

    # Pass output of layer two into activation function
    activation_Two.forward(layer_Two.output)

    # Calculate loss
    loss = loss_function.calculate(activation_Two.output, y)

    # Calculate accuracy from output of layer two activation
    predictions = np.argmax(activation_Two.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        '''
        If new model performance (loss) is better (smaller) than previous
            => copy weights and biases
        '''
        best_layerOne_weights = layer_One.weights.copy()
        best_layerOne_biases = layer_One.biases.copy()
        best_layerTwo_weights = layer_Two.weights.copy()
        best_layerTwo_biases = layer_Two.biases.copy()
        
        # Update lowest loss
        lowest_loss = loss

        # Update highest accuracy
        highest_accuracy = accuracy
        print(highest_accuracy)

    else:
        # Else copy previous best weights and biases for next iteration
        layer_One.weights = best_layerOne_weights.copy()
        layer_One.biases = best_layerOne_biases.copy()
        layer_Two.weights = best_layerTwo_weights.copy()
        layer_Two.biases = best_layerTwo_biases.copy()

print(f'---\nfinal accuracy: {round(highest_accuracy * 100, 2)}%')