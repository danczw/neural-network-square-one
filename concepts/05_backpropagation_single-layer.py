import numpy as np

'''
Neural Network backpropagation
    - calculating the impact of variables (weights & biases) on model's loss
    - demonstrated on a single layer with 3 neurons and input size of 4
    - reading up on derivates and chain rule recommended !
'''

# Passed gradient (derivates) from next layer, for demonstration purposes
dvalues = np.array([[1.0, 1.0, 1.0]])

# 4 weights per each of 4 input, for each of 3 neurons (transposed) 
w = np.array([[0.2, 0.8, -0.5, 1],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]).T

'''
Weights array is formatted so that rows contain weights related to each input
    => weights for all neurons for the given input
'''
print('transposed weights array:\n', w)

'''
Gradient with respect to input
    - sum weights related to the input
        (same index in each w array belongs to same input value over all neurons)
    - multiplied by gradient related to the neuron
        (corresponding neuron gradient for each input)
'''
di0 = sum([w[0][0] * dvalues[0][0],
           w[0][1] * dvalues[0][1],
           w[0][2] * dvalues[0][2]])

di1 = sum([w[1][0] * dvalues[0][0],
           w[1][1] * dvalues[0][1],
           w[1][2] * dvalues[0][2]])

di2 = sum([w[2][0] * dvalues[0][0],
           w[2][1] * dvalues[0][1],
           w[2][2] * dvalues[0][2]])

di3 = sum([w[3][0] * dvalues[0][0],
           w[3][1] * dvalues[0][1],
           w[3][2] * dvalues[0][2]])

'''
Simplification I of previous step due to Numpy array type
'''
di0 = sum(w[0]*dvalues[0])
di1 = sum(w[1]*dvalues[0])
di2 = sum(w[2]*dvalues[0])
di3 = sum(w[3]*dvalues[0])    

# Gradient with respect to input
dinputs = np.array([di0, di1, di2, di3])

'''
Simplification II of previous (two) steps due to Numpy array type

Note:
    - for numpy dotproduct, neighbouring dimensions need to match
    - shape of dvalues: (1, 3)
    - shape of (currently transposed!) weights (4, 3)
        => (1, 3) * (4, 3) = x
    - therefore, weights needs to be transposed => (3, 4)
        => (1, 3) * (3, 4) = ✓
'''
# Gradient with respect to input
dinputs = np.dot(dvalues[0], w.T)

print(dinputs, ' (gradient of the neuron function with respect to inputs)\n')

'''
Neural Network backpropagation with batch processing 
'''
# Passed gradient from next layer, for demonstration purposes
dvalues_batch = np.array([[1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.]])

# Batch input - 3 sets, for demonstration purposes
inputs_batch = np.array([[1., 2., 3., 2.5],
                          [2., 5., -1., 2.],
                          [-1.5, 2.7, 3.3, -0.8]])

# Batch output - 3 sets with one for each neuron, for demonstration purposes
outputs_batch = np.array([[1., 2., -3., -4.],
                          [2., -7., -1., 3.],
                          [9., 10., 11., 12.]])

# Biases - one bias for each of the 3 neurons, row vector with shape of (1, neurons)
biases = np.array([[2., 3., 0.5]])

'''
Input gradient with respect to weights
    - sum weights related to the input
        (same index in each w array belongs to same input value over all neurons)
    - multiplied by passed-in gradient related to the neuron
        (corresponding neuron gradient for each input)
'''
dinputs_batch = np.dot(dvalues_batch, w.T)
print(dinputs_batch, ' (gradient of the neuron function with respect to batch inputs)')

'''
Weights gradient with respect to inputs
    - inputs multiplied by passed-in gradient related to the neuron
        (corresponding neuron gradient for each weight)
'''
dweights_batch = np.dot(inputs_batch.T, dvalues_batch)
print(dweights_batch, ' (gradient of the neuron function with respect to batch weights)')

'''
Bias gradient with respect to passed-in gradient from previous layer
    - biases related to the inputs multiplied by passed-in gradient
    - related to the neuron (corresponding neuron gradient for each input)
'''
dbiases = np.sum(dvalues_batch, axis=0, keepdims=True) # keepdims keeps the gradient as row vector
print(dbiases, ' (gradient of the neuron function with respect to biases)')

'''
Activation function derivative - README.md on mathematical concept
'''
# Gradient of subsequent function, for demonstration purposes
dvalues_batch = np.array([[1., 2., 3., 4.],
                          [5., 6., 7., 8.],
                          [9., 10., 11., 12.]])

'''
Activation function (ReLU) gradient with respect to layer output
'''
# Array filled with zeros with shape of outputs
drelu = np.zeros_like(outputs_batch)

# Set values related to the inputs greater than 0 as 1
drelu[outputs_batch > 0] = 1

# Chain rule
drelu *= dvalues_batch

'''
Simplification of previous step
    - ReLU() derivative array is filled with 1s, which do not change the multiplies
        and 0s, which zero the multiplying value
    => take gradients of subsequent function and set to 0 all that are <=0
'''
drelu = dvalues_batch.copy()
drelu[dvalues_batch <= 0] = 0
print(drelu, ' (gradient of the activation function with respect to subsequent functions gradiant)')