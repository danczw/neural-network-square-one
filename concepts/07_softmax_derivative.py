'''
Softmax activation functions derivative
- partial derivative of the Softmax function
- more complicated than the derivative of the Categorical Cross-Entropy loss
- README.md for mathematical concept, code implementation below

np.eye():
- given a number, n, returns an nxn array filled with ones on the diagonal and zeros everywhere else
    > print(np.eye(5))
    > array([[1., 0., 0., 0., 0.],
    >        [0., 1., 0., 0., 0.],
    >        [0., 0., 1., 0., 0.],
    >        [0., 0., 0., 1., 0.],
    >        [0., 0., 0., 0., 1.]])
'''

import numpy as np

# Sample softmax_output
softmax_output = [0.7, 0.1, 0.2]

# Reshape to vector
softmax_output = np.array(softmax_output).reshape(-1, 1)

print(softmax_output.shape)

# Kronecker delta of softmax_output
print(np.eye(softmax_output.shape[0]), ' (Kronecker delta)')

# Derivative of softmax demands multiplication of output with Kronecker delta
print(softmax_output * np.eye(softmax_output.shape[0]), ' (multiplication as per derivative complex)')

# Previous multiplication can be simplified
print(np.diagflat(softmax_output), ' (multiplication as per derivative simplified)')

'''
Jacobian matrix:
- in this case an array of partial derivatives in all of the combinations
    of both input vectors
- calculating the partial derivatives of every output of the Softmax function
    with respect to each input separately, because each input influences each
    output due to the normalization process, which takes the sum of all the
    exponentiated inputs
- result of this operation, if performed on a batch of samples, is a list of
    the Jacobian matrices, which effectively forms a 3D matrix
- because each input influences all of the outputs, the returned vector of
    the partial derivatives has then to be summed up for the final
    partial derivative with respect to this input
'''
# Perform subtraction of both arrays resulting in a Jacobian matrix
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T), \
    ' (Jacobian matrix with array of partial derivatives of softmax)')
