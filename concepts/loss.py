'''
Calculating Loss with Categorical Cross-Entropy
    - abc
'''

import math

softmax_output = [0.7, 0.1, 0.2] # fictional output of softmax activation function of output layer
target_output = [1, 0, 0] # fictional target output
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)