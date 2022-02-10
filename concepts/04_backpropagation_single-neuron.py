'''
Neural Network backpropagation
    - calculating the impact of variables (weights & biases) on model's loss
    - demonstrated on a single neuron
'''

i = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weight for each input
b = 1.0 # bias

'''
forward pass
'''
# 1st step: weighting inputs
wi0 = i[0] * w[0] # weighted input one
wi1 = i[1] * w[1] # weighted input two
wi2 = i[2] * w[2] # weighted input three

print(wi0, wi1, wi2, '          (original weighted inputs)')

# 2nd step: adding weighted inputs and bias
no = wi0 + wi1 + wi2 + b # neuron output
print(no, '                   (original neuron output)')

# 3rd step: ReLU activation function
y = max(no, 0)
print(y, '                   (original loss)')

'''
backpropagation
    - how much does each input, weight and bias impact the output, i.e. the loss?
    - forward pass can be writen as:
        output =
            max(
                (inputs[0] * weights[0])
                + (inputs[1] * weights[1])
                + (inputs[2] * weights[2])
                + bias
            , 0)
    - derivates are established in order from outer to inner functions 

    => using chain rule and (partial) derivatives to calculate impact
        - calculate derivative of loss function
        - use it to multiply with the derivative of the activation function
            of the output layer
        - use result to multiply by derivative of output layer, and so on,
            through all of the hidden layers and activation functions
        - inside these layers, derivative with respect ot weights and biases
            will form gradients that are used to update weights and biases
        - derivatives with respect to inputs will form gradient to chain
            with previous layer which can caulcate the impact of its weights
            and biases on the loss and backpropagate gradients on inputs further

Note:
    - d in front of variable for derivative
    - da_db => derivative of function a in respect to var b 
'''
# for demonstration, neuron receives gradient of 1 from next layer
dvalue = 1.0

# derivative of ReLU (3rd step in forward pass) (drelu_dno) with respect to
# its input (dno) is 1 if input is greater than 0, else 0, then using chain rule
drelu_dno = dvalue * (1.0 if no > 0 else 0.0)
print(drelu_dno, '                   (drelu_dno)')

# partial derivative of ReLU with respect to weighted inputs (2nd step in forward pass) 
# Note: derivative of sum operation is always 1 in this case
dsum_dwi0 = 1 # partial derivative of the sum (var 'no') with respect to the input for the first pair of inputs and weights
drelu_dwi0 = drelu_dno * dsum_dwi0 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is ReLU

dsum_dwi1 = 1 # partial derivative of the sum (var 'no') with respect to the input for the second pair of inputs and weights
drelu_dwi1 = drelu_dno * dsum_dwi1 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is ReLU

dsum_dwi2 = 1 # partial derivative of the sum (var 'no') with respect to the input for the third pair of inputs and weights
drelu_dwi2 = drelu_dno * dsum_dwi2 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is ReLU

dsum_db = 1 # partial derivative of the sum (var 'no') with respect to the input for the bias
drelu_db = drelu_dno * dsum_db # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is ReLU

print(drelu_dwi0, drelu_dwi1, drelu_dwi2, drelu_db, '       (drelu_dwi1, drelu_dwi2, drelu_dwi3, drelu_db)')

# continue backwards to function before the sum (1st step in forward pass) is weighting of inputs
dwi_di0 = w[0] # partial derivative of the multiplication (var 'wi0') with respect to the first input
drelu_di0 = drelu_dwi0 * dwi_di0 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

dwi_dw0 = i[0] # partial derivative of the multiplication (var 'wi0') with respect to the first weight
drelu_dw0 = drelu_dwi0 * dwi_dw0 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

dwi_di1 = w[1] # partial derivative of the multiplication (var 'wi1') with respect to the second input
drelu_di1 = drelu_dwi1 * dwi_di1 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

dwi_dw1 = i[1] # partial derivative of the multiplication (var 'wi1') with respect to the second weight
drelu_dw1 = drelu_dwi1 * dwi_dw1 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

dwi_di2 = w[2] # partial derivative of the multiplication (var 'wi2') with respect to the third input
drelu_di2 = drelu_dwi2 * dwi_di2 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

dwi_dw2 = i[2] # partial derivative of the multiplication (var 'wi2') with respect to the third weight
drelu_dw2 = drelu_dwi2 * dwi_dw2 # chain rule - multiplying partial derivative with the derivative of the subsequent function, which is partial derivative of ReLU

print(drelu_di0, drelu_dw0, drelu_di1, drelu_dw1, drelu_di2, drelu_dw2, \
    '             (drelu_di1, drelu_dw1, drelu_di2, ...)')

'''
simplification of previous steps on example of first input
    drelu_di0 = drelu_dwi0 * dwi_di0            # where dwi_di0 = w[0]
    drelu_di0 = drelu_dwi0 * w[0]               # where drelu_dwi0 = drelu_dno * dsum_dwi0
    drelu_di0 = drelu_dno * dsum_dwi0 * w[0]    # where dsum_dwi0 = 1
    drelu_di0 = drelu_dno * 1 * w[0]            # where drelu_dno = dvalue * (1.0 if no > 0 else 0.0)
    drelu_di0 = dvalue * (1.0 if no > 0 else 0.0) * w[0]
'''

# partial derivatives combined into a vector make up gradients
di = [drelu_di0, drelu_di1, drelu_di2] # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradients on weights
db = drelu_db # gradient on bias - example includes only one bias

print(di, dw, db, ' (input gradient, weight gradient, bias gradient)')

'''
apply fraction of gradients to values to decrease output (loss)

Note:
    - di excluded, as there is only a single neuron in a single layer
    - with preceding layers, partial derivative with respect to
        inputs would be calculated
'''
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b, '          (adjusted weights, bias)')

'''
effects to be viewed through another forward pass
'''
# weighting inputs
wi0 = i[0] * w[0] # weighted input one
wi1 = i[1] * w[1] # weighted input two
wi2 = i[2] * w[2] # weighted input three

print(wi0, wi1, round(wi2,3), ' (adjusted weighted inputs)')

# adding weighted inputs and bias
no = wi0 + wi1 + wi2 + b # neuron output
print(no, '              (adjusted neuron output)')

# ReLU activation function
y = max(no, 0)
print(y, '              (adjusted loss)')
