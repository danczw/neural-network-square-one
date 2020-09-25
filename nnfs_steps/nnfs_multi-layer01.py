# multi layer calculation of a Neural Network with batch input

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights_One = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

biases_One = [2.0, 3.0, 0.5]

# added a second layer with sets of weights and biases
weights_Two = [[0.1, -0.14, 0.5],
               [0.5, 0.12, -0.33],
               [-0.44, 0.73, -0.13]]

biases_Two = [-1.0, 2.0, -0.5]

'''
"manual" calculation adding a second layer
output of layer one is input of layer two with its own weights and biases
'''

layer1_outputs = np.dot(inputs, np.array(weights_One).T) + biases_One

# layer 1 outputs become inputs for layer two
layer2_outputs = np.dot(layer1_outputs, np.array(weights_Two).T) + biases_Two

print(layer2_outputs)