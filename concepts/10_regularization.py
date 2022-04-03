'''
L1 and L2 Regularization in forward pass
    - methods to reduce generalization error
    - L1 and L2 regularization are used to calculate a penalty added to the loss
        value to penalize model for large weights and biases
    => large weights and biases might indicate that a neuron is attempting to
        memorize a data element instead of generalization
    - can also help in case of exploding gradients: model instability, which
        might cause weights to beckome very large values
    - L1 regularization
        - sum of all the absolute values for weights and biases
        - linear penalty as regularization loss returned by this function is
            directly proportional to parameter values
        - penalizes small values more than L2, causing the model to start being
            invariant to small inputs and variant only the bigger once
        - rarely used alone and usually combined with L2 regularization if at all
    - L2 regularization
        - sum of squared weights and biases
        - non-linear penatly as regularization loss returned by this function
            penalizes larger weights and biases more than smaller ones
        - L2 commonly used as it does not affect smaller parameter values
            substantially and does not allow model to grow weights and biases
            too large by heavily penalizing relatively big values
    - value Lambda is used, where higher value means more significant penalty
'''

import numpy as np

# Weights and bias of one neuron
weights = np.array([0.2, 0.8, -0.5])
biases = np.array([0.9])

# Lambda values for L1 and L2 regularization, each for weights and bias
lambda_l1w = 0.1
lambda_l2w = 0.2
lambda_l1b = 0.3
lambda_l2b = 0.4

# Fictious data loss (loss function)
data_loss = 0.55

# Forward pass of L1 and L2 regularization
l1w = lambda_l1w * sum(abs(weights))
l1b = lambda_l1b * sum(abs(biases))
l2w = lambda_l2w * sum(weights**2)
l2b = lambda_l2b * sum(biases**2)
loss = data_loss + l1w + l1b + l2w + l2b

print(f'Loss: {loss:.3f}, L1w: {l1w:.3f}, L1b: {l1b:.3f}, L2w: {l2w:.3f}, L2b: {l2b:.3f}')

'''
Backward pass of L1 and L2 regularization
    - L1 derivative effectively multiplies value by -1 if it is less than 0,
        otherwise, it's multiplied by 1
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

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

dl1 = np.ones_like(weights)
dl1[weights < 0] = -1

print(dl1)