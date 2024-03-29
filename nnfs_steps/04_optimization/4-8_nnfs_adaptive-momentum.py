import matplotlib.pyplot as plt
import numpy as np

# Import basic dataset
import nnfs
nnfs.init()

def main():
    '''
    Optimization of Neural Network weights and biases
        - added root mean square propagation optimization
            for each parameter (weights and biases)
    '''

    # TODO: choose between 'vertical' and 'spiral' dataset
    dataset = 'spiral'

    if dataset == 'vertical':
        # Import vertical dataset
        from nnfs.datasets import vertical_data
        X, y = vertical_data(samples=100, classes=3)

        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
        plt.show()
    elif dataset == 'spiral':
        # Import spriral dataset
        from nnfs.datasets import spiral_data
        X, y = spiral_data(samples=100, classes=3)

        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
        plt.show()

    # Define class to initialize layer
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            # Shape of weights array based on input shape and number of neurons
            self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
            
            # Shape of biases based on number of neurons, initial biases are set to 0
            self.biases = np.zeros((1, n_neurons))

        # For first layer, input is actual input data (X), every other layer input is self.output of prev layer
        def forward(self, inputs):
            # Remember inputs for creating derivatives during backpropagation
            self.inputs = inputs
            # Output is dotproduct + biases calc
            self.output = np.dot(inputs, self.weights) + self.biases

        def backward(self, dvalues):
            # Gradient of weights with respect to inputs
            self.dweights = np.dot(self.inputs.T, dvalues)
            
            # Graident of biases with respect to passed-in gradients
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
            
            # Graident of inputs with respect to weights
            self.dinputs = np.dot(dvalues, self.weights.T)

    # Define class to initialize activation function: rectified linear unit
    class Activation_ReLU:
        def forward(self, inputs):
            # Remember inputs for creating derivatives during backpropagation
            self.inputs = inputs
            self.output = np.maximum(0, inputs)

        def backward(self, dvalues):
            self.dinputs = dvalues.copy()     

            # Zero gradient where input values were negative                              
            self.dinputs[self.inputs <= 0] = 0

    # Define class to initialize activation function: Softmax
    class Activation_Softmax:
        def forward(self, inputs):
            # Remember input values
            self.inputs = inputs

            # Get propabilities, minus np.max to prevent overflow problem
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            
            # Normalize propabilities
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            
            self.output = probabilities
        
        def backward(self, dvalues):
            # Create uninitialized array with same shape as dvalues
            self.dinputs = np.empty_like(dvalues)

            for index, (single_output, single_dvalue) in enumerate(zip(self.output,
                                                                    dvalues)):
                # Flatten output array
                single_output = single_output.reshape(-1, 1)

                # Calculate Jacobin matrix of the output
                jacobin_matrix = np.diagflat(single_output) - np.dot(single_output,
                                                                    single_output.T)
                
                # Calculate sample-wise gradient
                self.dinputs[index] = np.dot(jacobin_matrix, single_dvalue)

    # Define class to initialize loss function
    class Loss:
        # Calculate mean loss and pass forward
        def calculate(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

    # Define class to initialize categorical cross entropy
    class Loss_CategoricalCrossentropy(Loss):
        # Calculate categorical cross entropy
        def forward(self, y_pred, y_true):
            samples = len(y_pred)

            # Clip y_pred to prevent inf loss when calculating loss of y_pred = 0
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

            # Dynamicaly handel confidences for different target var formatting
            if len(y_true.shape) == 1: # scalar / categorical values
                correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape) == 2: # one-hot-encoded
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

            negative_log_likelihoods = -np.log(correct_confidences)
            
            return negative_log_likelihoods

        def backward(self, dvalues, y_true):
            # Number of samples
            samples = len(dvalues)

            # Number of labels as per first sample
            labels = len(dvalues[0])
            
            # Turn numerical labels into one-hot encoded vectors if labels are sparse
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]

            # Calculate gradient
            self.dinputs = -y_true / dvalues

            # Normalization of values by calculating their mean
            self.dinputs = self.dinputs / samples

    # Softmax classifier - combined Softmax activation and cross-entropy loss function
    class Activation_Softmax_Loss_CategoricalCrossEntropy():
        # Create activation and loss function objects
        def __init__(self):
            self.activation = Activation_Softmax()
            self.loss = Loss_CategoricalCrossentropy()

        # Forward pass
        def forward(self, inputs, y_true):
            # Output layer's activation function
            self.activation.forward(inputs)

            # Set the output
            self.output = self.activation.output

            # Calculate and return loss value
            return self.loss.calculate(self.output, y_true)

        # Backward pass
        def backward(self, dvalues, y_true):
            # Number of samples
            samples = len(dvalues)

            # Turn one-hot encoded vectors into numerical labels
            if len(y_true.shape) == "":
                y_true = np.argmax(y_true, axis=1)
            
            # Copy to safely modify
            self.dinputs = dvalues.copy()

            # Calculate gradient
            self.dinputs[range(samples), y_true] -= 1

            # Normalize gradients
            self.dinputs = self.dinputs / samples

    class Optimizer_SGD:
        def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            # View concepts/09_learning_rate.py for more information
            self.decay = decay
            self.iterations = 0
            # Added momentum to update_params function
            self.momentum = momentum
        
        # Call before any parameters are updated
        def pre_update_params(self):
            # Update learning rate
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                                            (1.0 / (1.0 +
                                                    self.decay *
                                                    self.iterations))

        # Update parameters
        def update_params(self, layer):
            # If momentum is used
            if self.momentum:
                # If layer has no momentum arrays, create with 0s
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.biases_momentums = np.zeros_like(layer.biases)
                
                # Weight updates with momentum
                weight_updates = self.momentum * layer.weight_momentums - \
                                self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates

                # Bias updates with momentum
                bias_updates = self.momentum * layer.biases_momentums - \
                            self.current_learning_rate * layer.dbiases
                layer.biases_momentums = bias_updates
            # Else simple SGD updates without momentum
            else:
                weight_updates -= self.learning_rate * layer.dweights
                bias_updates -= self.learning_rate * layer.dbiases
            
            # Update weights and biases
            layer.weights += weight_updates
            layer.biases += bias_updates
        
        # Call after parameters are updated
        def post_update_params(self):
            # Update iteration counter
            self.iterations += 1

    class Optimizer_AdaGrad:
        def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            # View concepts/09_learning_rate.py for more information
            self.decay = decay
            self.iterations = 0
            # Added epsilon to AdaGrad update_params function
            self.epsilon = epsilon
        
        # Call before any parameters are updated
        def pre_update_params(self):
            # Update learning rate
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                                            (1.0 / (1.0 +
                                                    self.decay *
                                                    self.iterations))

        # Update parameters
        def update_params(self, layer):
            # If layer has no cache arrays (history of sqared gradients), create with 0s
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.biases_cache = np.zeros_like(layer.biases)
                
            # Update cache with squared current gradients
            layer.weight_cache += layer.dweights**2
            layer.biases_cache += layer.dbiases**2

            # Weight update and normalization with square root cache
            layer.weights += -self.current_learning_rate * layer.dweights \
                            / (np.sqrt(layer.weight_cache) + self.epsilon)

            # Bias update and normalization with square root cache
            layer.biases += -self.current_learning_rate * layer.dbiases \
                            / (np.sqrt(layer.biases_cache) + self.epsilon)
        
        # Call after parameters are updated
        def post_update_params(self):
            # Update iteration counter
            self.iterations += 1

    class Optimizer_RMSprop:
        def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            # View concepts/09_learning_rate.py for more information
            self.decay = decay
            self.iterations = 0
            # Added epsilon and rho to RMSprop update_params function
            self.epsilon = epsilon
            self.rho = rho
        
        # Call before any parameters are updated
        def pre_update_params(self):
            # Update learning rate
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                                            (1.0 / (1.0 +
                                                    self.decay *
                                                    self.iterations))

        # Update parameters
        def update_params(self, layer):
            # If layer has no cache arrays (history of sqared gradients), create with 0s
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.biases_cache = np.zeros_like(layer.biases)
                
            # Update cache with squared current gradients and retained part of cache
            layer.weight_cache = self.rho * layer.weight_cache + \
                                (1 - self.rho) * layer.dweights**2
            layer.biases_cache = self.rho * layer.biases_cache + \
                                (1 - self.rho) * layer.dbiases**2

            # Weight update and normalization with square root cache
            layer.weights += -self.current_learning_rate * layer.dweights \
                            / (np.sqrt(layer.weight_cache) + self.epsilon)

            # Bias update and normalization with square root cache
            layer.biases += -self.current_learning_rate * layer.dbiases \
                            / (np.sqrt(layer.biases_cache) + self.epsilon)
        
        # Call after parameters are updated
        def post_update_params(self):
            # Update iteration counter
            self.iterations += 1

    class Optimizer_Adam:
        '''
        Adaptive Momentum (Adam)
            - Most widely-used optimization algorithm and built atop RMSprop
            - Adds back in momentum concept from SGD
            => Instead of applying current gradients, applies momentums as in SGD,
                then apply per-weight adaptive learning rate with cache as done
                in RMSprop
            - Addionally, adds bias correction mechanism (beta)
                - Applied to cache and momentum, compensating for initially
                    zeroed values
                - Both momentum and cache are divided by 1-beta**step 
                - As step raises, beta**step approaches 0, solving whole expression
                    to a fraction during first steps and approaching 1 as
                    training progresses
        '''
        def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7,
                    beta_One=0.9, beta_Two=0.999):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            # View concepts/09_learning_rate.py for more information
            self.decay = decay
            self.iterations = 0
            # Added epsilon and betas to Adam update_params function
            self.epsilon = epsilon
            self.beta_One = beta_One
            self.beta_Two = beta_Two
        
        # Call before any parameters are updated
        def pre_update_params(self):
            # Update learning rate
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                                            (1.0 / (1.0 +
                                                    self.decay *
                                                    self.iterations))

        # Update parameters
        def update_params(self, layer):
            # If layer has no cache arrays, create with 0s
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

            '''
            Adding Adam
                - Combines RMSprop and SGD momentum
            '''
            # Update momentum with current gradients
            layer.weight_momentums = self.beta_One * layer.weight_momentums + \
                                    (1 - self.beta_One) * layer.dweights
            layer.bias_momentums = self.beta_One * layer.bias_momentums + \
                                (1 - self.beta_One) * layer.dbiases

            # Get corrected momentum
            weight_momentums_corrected = layer.weight_momentums / \
                                        (1 - self.beta_One ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / \
                                    (1 - self.beta_One ** (self.iterations + 1))
            
            # Update cache with squared current gradients
            layer.weight_cache = self.beta_Two * layer.weight_cache + \
                                (1 - self.beta_Two) * layer.dweights**2
            layer.bias_cache = self.beta_Two * layer.bias_cache + \
                            (1 - self.beta_Two) * layer.dbiases**2
            
            # Get corrected cache
            weight_cache_corrected = layer.weight_cache / \
                                    (1 - self.beta_Two ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / \
                                (1 - self.beta_Two ** (self.iterations + 1))
            
            
            # Parameter update and normalization with square root cache
            layer.weights += -self.current_learning_rate * \
                            weight_momentums_corrected / \
                            (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * \
                            bias_momentums_corrected / \
                            (np.sqrt(bias_cache_corrected) + self.epsilon)
        
        # Call after parameters are updated
        def post_update_params(self):
            # Update iteration counter
            self.iterations += 1

    '''
    Initialize layers and activation function using
        common Categorical Cross-Entropy loss and Softmax activation
    '''
    layer_One = Layer_Dense(2, 64)
    activation_One = Activation_ReLU()
    layer_Two = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
    # TODO: try different hyperparameter for Adam optimization
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

    # set epochs, i.e. loops on how often to optimize the parameter
    epochs = 10001

    for epoch in range(epochs):
        # Pass data through layers, original input is X
        layer_One.forward(X)

        # Pass output of layer one into activation function
        activation_One.forward(layer_One.output)

        # Pass output of activation one into layer two
        layer_Two.forward(activation_One.output)

        # Pass output of layer two into combined loss activation function
        loss = loss_activation.forward(layer_Two.output, y)

        # Get predictions by finding index of highest class confidence
        predictions = np.argmax(loss_activation.output, axis=1)

        # Convert targets if one-hot encoded
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predictions==y)

        if not epoch % 100:
            print((f'epoch: {epoch}, '
                f'loss: {loss:.3f}, '
                f'accuracy: {accuracy:.3f}, '
                f'learning rate: {optimizer.current_learning_rate:.3f}'))

        # Backpropagete outputs based on true class through loss and softmax function
        loss_activation.backward(loss_activation.output, y)

        # Backpropagate softmax function derivatives through layer two
        layer_Two.backward(loss_activation.dinputs)

        # Backpropagate layer two derivatives through activation one (relu)
        activation_One.backward(layer_Two.dinputs)

        # Backpropagate activation one (relu) derivatives through layer one
        layer_One.backward(activation_One.dinputs)

        # Update learning rate
        optimizer.pre_update_params()

        # Use SGD optimizer to update parameters
        optimizer.update_params(layer_One)
        optimizer.update_params(layer_Two)

        # Update iteration counter
        optimizer.post_update_params()

if __name__ == '__main__':
    main()