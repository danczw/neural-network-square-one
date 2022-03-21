'''
Backward pass of L1 and L2 regularization
    - L1 derivative effectively multiplies value by -1 if it is less than 0,
        otherwise, it'S multiplied by 1
    - this is due to the absolute function being linear for positive values,
        which derivative equals 1
    - for negative values, it negates the sign of the value to make it
        positive, resulting in the derivative being -1
    - see README.md for mathematical details on L1 and L2 derivative
'''
# Weight of one neuron
weights = [0.2, 0.8, -0.5]

# Array of partial derivative of L1 regularization
dl1 = []

for weight in weights:
    if weight >= 0:
        dl1.append(1)
    else:
        dl1.append(-1)
print(dl1)

'''
Implementation for multi-neuron layer
'''
# 3 sets of weights
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Array of partial derivative of L1 regularization
dl1 = []

for neuron in weights:
    # Derivatives related to one neuron
    neuron_dl1 = []
    
    for weight in neuron:
        if weight >= 0:
            neuron_dl1.append(1)
        else:
            neuron_dl1.append(-1)
    dl1.append(neuron_dl1)

print(dl1)

'''
Implementation with Numpy
'''

import numpy as np

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

dl1 = np.ones_like(weights)
dl1[weights < 0] = -1

print(dl1)