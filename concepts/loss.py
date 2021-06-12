'''
Calculating Neural Network Loss with Categorical Cross-Entropy
    - loss as measurement of error
    - with higher confidence in prediction (softmax_output) loss is lower and vice versa 
'''

import math

softmax_output = [0.7, 0.1, 0.2]    # fictional output of softmax activation function of output layer
target_output = [1, 0, 0]           # fictional target output
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss) # loss of current model


'''
Concept for easier target variable selection when working with arrays
'''

import numpy as np

softmax_output_array = np.array([[0.7, 0.1, 0.2],
                                 [0.1, 0.5, 0.4],
                                 [0.02, 0.9, 0.08]])
class_target = [0, 1, 1]

print(softmax_output_array[
    range(len(softmax_output_array)),   # iterate through array
    class_target                        # get each target class
]) # selects the target class for each observation of the output array