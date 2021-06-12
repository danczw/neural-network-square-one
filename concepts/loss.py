'''
Calculating Neural Network Loss with Categorical Cross-Entropy
    - loss as measurement of error
    - with higher confidence in prediction (softmax_output) loss is lower and vice versa 
'''

import math

softmax_output = [0.7, 0.1, 0.2] # fictional output of softmax activation function of output layer
target_output = [1, 0, 0] # fictional target output
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss) # loss of current model