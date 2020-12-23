'''
simple single layer calculation of a
Neural Network without activation function
'''
inputs = [1.0, 2.0, 3.0, 2.5] # 4 input values
biases = [2.0, 3.0, 0.5] # biasis for three neurons
# 4 input values á 3 neurons => 12 weights, 4 per neuron
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
layer_outputs = [] # output of current layer

'''
output calculation for each of the three neuron:
    output[0] = weight[0][0] * input[0] + weight[0][1] * input[1] ... + bias
    output[1] = weight[1][0] ...
    output[2] = ...
using zip() for dotproduct calc as above
'''
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # initialise output of given neuron
    for neuron_input, weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)