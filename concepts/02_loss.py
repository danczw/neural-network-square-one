'''
Calculating Neural Network Loss with Categorical Cross-Entropy
    - calculation of loss using single observation with 3 possible classes
    - loss as measurement of error
    - with higher confidence in prediction (softmax_output) loss is lower and vice versa 
'''

import math

target_output = [1, 0, 0]           # fictional target output: first class is correct
softmax_output = [0.7, 0.1, 0.2]    # fictional output of softmax activation function of output layer for each class - confidence per class
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss) # loss of current model


'''
Concept for easier target variable selection when working with arrays
'''

import numpy as np

class_target = [0, 1, 1]
softmax_output_array = np.array([[0.7, 0.1, 0.2],
                                 [0.1, 0.5, 0.4],
                                 [0.02, 0.9, 0.08]]) # confidence per class

print(softmax_output_array[
    range(len(softmax_output_array)),   # iterate through array
    class_target                        # get each target class
]) # selects the target class for each observation of the output array

'''
Calculating Neural Network Loss with Categorical Cross-Entropy
    - calculation of loss using array 
'''

class_target = [0, 1, 1]
# caution: prediction for class target of first observation is 0.0
#   => this would result in infinite loss, as log of 0.0 is infinite
softmax_output_array = np.array([[0.0, 0.1, 0.2],       
                                 [0.1, 0.5, 0.4],
                                 [0.02, 0.9, 0.08]])
# therefore, prediction is clipped to circumvent inf problem
softmax_output_array_clip = np.clip(softmax_output_array, 1e-7, 1 - 1e-7)
print(softmax_output_array_clip)

print(
    np.mean(                                            # mean loss of all observations
        -np.log(                                        # similar to -(math.log(...) + ...) but for arrays                  
            softmax_output_array_clip[
                range(len(softmax_output_array_clip)),
                class_target
            ]
        )
    )
)