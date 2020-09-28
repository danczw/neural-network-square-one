# single layer calculation of a Neural Network with batch input

import numpy as np

# changed to batch input of [3,4]
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

'''
for dotproduct of batch input one array needs to be transposed
- in this case done via transpose of weights array using numpy arrays functions
'''

output = np.dot(inputs, np.array(weights).T) + biases

print(output)