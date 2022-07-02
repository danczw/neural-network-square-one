import numpy as np

def main():
    '''
    Multi layer calculation of a Neural Network with batch input
    '''

    # Changed to batch input of [3,4]
    inputs = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

    # Biasis for three neurons in first layer
    biases_One = [2.0, 3.0, 0.5]

    # 4 input values á 3 neurons => 12 weights, 4 per neuron in first layer
    weights_One = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

    # Biasis for three neurons in second layer
    biases_Two = [-1.0, 2.0, -0.5]

    # Added a second layer with 3 neurons and respective sets of weights and biases
    #   => 3 Neurons in previous layer result in 3 inputs for current layer
    weights_Two = [[0.1, -0.14, 0.5],
                [0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]

    '''
    Manual calculation adding a second layer of 3 neurons output of layer one
        is input of layer two with its own weights and biases
    '''
    layer1_outputs = np.dot(inputs, np.array(weights_One).T) + biases_One

    # Layer 1 outputs become inputs for layer two
    layer2_outputs = np.dot(layer1_outputs, np.array(weights_Two).T) + biases_Two

    '''
    Output [3,3] corresponds to one output value per neuron in
        last layer (3 neurons) per batch (3 batches)
    '''
    print(layer2_outputs)

if __name__ == '__main__':
    main()