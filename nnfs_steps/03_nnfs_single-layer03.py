'''
simple single layer calculation of a
Neural Network without activation function
    - batch input version
'''
import numpy as np

# changed to batch input of [3,4]
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
biases = [2.0, 3.0, 0.5] # biasis for three neurons
# 4 input values á 3 neurons => 12 weights, 4 per neuron
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

'''
output calc for each of the three neuron:
    output[0] = weight[0][0] * input[0] + weight[0][1] * input[1] ... + bias
    output[1] = weight[1][0] ...
    output[2] = ...
using np.dot() for dotproduct calculation as above

Note: size at index 1 of first argument array needs to match size at index 0
    of second argument array
= > for dotproduct of batch input one array needs to be transposed
    - in this case done via transpose of weights array using numpy arrays functions
'''

output = np.dot(inputs, np.array(weights).T) + biases

'''
output [3,3] corresponds to one ouput value per neuron (3 neurons)
per batch (3 batches)
'''
print(output)