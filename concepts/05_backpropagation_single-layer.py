'''
Neural Network backpropagation
    - calculating the impact of variables (w & biases) on model's loss
    - demonstrated on a single layer
'''

import numpy as np

# passed gradient from next layer, for demonstration purposes
dvalues = np.array([[1.0, 1.0, 1.0]])

# 4 weights per each of the 4 input, for each of 3 neurons (transposed) 
w = np.array([[0.2, 0.8, -0.5, 1],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]).T

# weights array is formatted so that rows contain weights related to each input
# -> weights for all neurons for the given input
print('transposed weights array:\n', w)

# Sum weights related to the input (same index in each w array belongs to same input value over all neurons)
# multiplied by gradient related to the neuron (corresponding neuron gradient for each input)
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
simplification of previous step due to Numpy array type
'''
di0 = sum(w[0]*dvalues[0])
di1 = sum(w[1]*dvalues[0])
di2 = sum(w[2]*dvalues[0])
di3 = sum(w[3]*dvalues[0])    


dinputs = np.array([di0, di1, di2, di3])

'''
simplification of previous (two) steps due to Numpy array type

Note:
    - for numpy dotproduct, neighbouring dimensions need to match
    - shape of dvalues: (1, 3)
    - shape of (currently transposed!) weights (4, 3)
        => (1, 3) * (4, 3) = x
    - therefore, weights needs to be transposed => (3, 4)
        => (1, 3) * (3, 4) = ✓
'''
dinputs = np.dot(dvalues[0], w.T)

print(dinputs, ' (gradient of the neuron function with respect to inputs)')

'''
Neural Network backpropagation
with batch processing 
'''
# passed gradient from next layer, for demonstration purposes
dvalues_batch = np.array([[1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.]])

# Sum weights related to the input (same index in each w array belongs to same input value over all neurons)
# multiplied by gradient related to the neuron (corresponding neuron gradient for each input)
dinputs_batch = np.dot(dvalues_batch, w.T)
print(dinputs_batch, ' (gradient of the neuron function with respect to batch inputs)')