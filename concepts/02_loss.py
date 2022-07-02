import math
import numpy as np

def basic_loss():
    '''
    Calculating Neural Network Loss with Categorical Cross-Entropy
        - calculation of loss using single observation with 3 possible classes
        - loss as measurement of error
        - with higher confidence in prediction (softmax_output) loss is lower
            and vice versa 
    '''

    # Fictional target output: first class is correct
    target_output = [1, 0, 0]

    # Fictional output of softmax activation function of output layer for each class
    #   => confidence per class
    softmax_output = [0.7, 0.1, 0.2]

    loss = -(math.log(softmax_output[0]) * target_output[0] +
            math.log(softmax_output[1]) * target_output[1] +
            math.log(softmax_output[2]) * target_output[2])

    # Loss of current model
    print(loss)

def target_var():
    '''
    Concept for easier target variable selection when working with arrays
    '''

    class_target = [0, 1, 1]

    # Confidence per class for 3 inputs
    softmax_output_array = np.array([[0.7, 0.1, 0.2],
                                    [0.1, 0.5, 0.4],
                                    [0.02, 0.9, 0.08]])

    # Selects the target class for each observation of the output array
    print(softmax_output_array[
        # Iterate through array
        range(len(softmax_output_array)),

        # Get each target class
        class_target
    ])

def array_loss():
    '''
    Calculating Neural Network Loss with Categorical Cross-Entropy
        - calculation of loss using array 
    '''
    class_target = [0, 1, 1]

    # Caution: prediction for class target of first observation is 0.0
    #   => this would result in infinite loss, as log of 0.0 is infinite
    softmax_output_array = np.array([[0.0, 0.1, 0.2],       
                                    [0.1, 0.5, 0.4],
                                    [0.02, 0.9, 0.08]])

    # Therefore, prediction is clipped to circumvent infinite problem
    softmax_output_array_clip = np.clip(softmax_output_array, 1e-7, 1 - 1e-7)
    print(softmax_output_array_clip)

    print('loss:',
        # Mean loss of all observations
        np.mean(
            # Similar to -(math.log(...) + ...) but for arrays
            -np.log(                  
                softmax_output_array_clip[
                    range(len(softmax_output_array_clip)),
                    class_target
                ]
            )
        )
    )

if __name__ == '__main__':
    basic_loss()
    target_var()
    array_loss()