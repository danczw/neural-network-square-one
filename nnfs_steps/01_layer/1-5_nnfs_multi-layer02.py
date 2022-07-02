import numpy as np

def main():
    '''
    Multi layer calculation of a Neural Network with batch input
        - added OOP
    '''

    # Changed input naming to X to conform with notation standards
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

    # Define class to initialize layer
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            '''
            Keep initital weights close to 0.1 to not create
                infinitively large number by later propagation through layers
            '''
            # Shape of weights array based on input shape and number of neurons
            self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
            
            # Shape of biases based on number of neurons, initial biases are set to 0
            self.biases = np.zeros((1, n_neurons))

        '''
        For first layer, input is actual input data (X),
            every other layer input is self.output of prev layer
        '''
        def forward(self, inputs):
            # Output is dotproduct + biases calculations
            self.output = np.dot(inputs, self.weights) + self.biases

    '''
    Initialize layers using layer classes with arguments of
        shape of input (features) and number of neurons
    '''
    layer_One = Layer_Dense(4, 5)
    # Second layer input corresponds with previous number of neurons
    #   => therefore pervious output shape
    layer_Two = Layer_Dense(5, 2)

    # Pass data through layers, original input is X
    layer_One.forward(X)

    print(layer_One.output)

    # Second layer input is first layer output
    layer_Two.forward(layer_One.output)

    print(layer_Two.output)

if __name__ == '__main__':
    main()