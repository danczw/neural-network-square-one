'''
Stochastic Gradient Descent (SGD) as Optimization Algorithm
    - use gradients to adjust parameters (i.e. weights and biases)
        to decrease measure of loss
    - follows direction of steepest descent of the gradient at a given point
    - choose a learning rate, which then subtracts the
        learning_rate * parameter_gradient from actual parameter values
    - thereby optimizing model parameter based on gradient of very last
        model function (due to backpropagation): the loss functions
'''
# SGD optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.bias -= self.learning_rate * layer.dbias
