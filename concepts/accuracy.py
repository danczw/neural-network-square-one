'''
Calculating Neural Network accuracy
    - calculation of accuracy by comparing class target with prediction
    - describes how often the largest confidence is the correct class
      in terms of a fraction
'''

import numpy as np

class_target = [0, 1, 1]
softmax_output_array = np.array([[0.7, 0.1, 0.2],
                                 [0.5, 0.1, 0.4],
                                 [0.02, 0.9, 0.08]]) # confidence per class

# get predictions by finding index of highest class confidence
predictions = np.argmax(softmax_output_array, axis=1)

print(f'predictions: {predictions}')

# accuracy is the mean difference between predicted and 'real' (class_target) value
accuracy = np.mean(predictions == class_target)

print(f'acc: {accuracy}')