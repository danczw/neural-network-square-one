'''
Neural Network backpropagation with batch processing and optimization
    - calculating the impact of variables (w & biases) on model's loss
    - demonstrated on a single layer with 3 neurons and input size of 4
    - reading up on derivates recommended
'''

import numpy as np

# Passed gradient from next layer, for demonstration purposes
dvalues_batch = np.array([[1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.]])

# Batch input - 3 sets á 4 inputs, for demonstration purposes
inputs_batch = np.array([[1., 2., 3., 2.5],
                          [2., 5., -1., 2.],
                          [-1.5, 2.7, 3.3, -0.8]])

# Weights - 4 weights per each of 4 inputs, for each of 3 neurons (transposed) 
weights = np.array([[0.2, 0.8, -0.5, 1],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]).T

# Biases - one bias for each of the 3 neurons, row vector with shape of (1, neurons)
biases = np.array([[2., 3., 0.5]])

'''
Forward pass
'''
# Dense layer
layer_outputs = np.dot(inputs_batch, weights) + biases

# ReLu activation
relu_outputs = np.maximum(0, layer_outputs)

'''
Backpropagation
'''
# ReLU activation - simulates gradient with respect to input values from
#   next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense Layer
# Gradient of inputs with respect to weigths - multiply drelu by weights
dinputs = np.dot(drelu, weights.T)
# Gradient of weights with respect to inputs - multiply drelu by inputs
dweights = np.dot(inputs_batch.T, drelu)
# Gradient with respect to bias
dbiases = np.sum(drelu, axis=0, keepdims=True) # keepdims keeps the gradient as row vector

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights, ' (updated weights)')
print(biases, ' (updated biases)')