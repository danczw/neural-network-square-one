'''
Learning Rate
    - negative gradient values usually too big to be applied to parameters
        (as seen in 08_stochastic_gradient_descent.py)
    - instead, use small steps to adjust weights and biases by using a fraction
        (negative) of the gradient value
    - small steps assure to follow direction of deepest descent, but can be too
        small to be effective and cause learning stagnation
    - this can be due to a local minimum, i.e. a local minimum of the loss function,
        which might not be the global minumum and thereby not best optimization
    - Solution: introducing inertia to the optimization process
        - size of inertia can speed up optimization to find global minimum
        - but too much inertia can cause problems, e.g. "jumping out" of a
            otherwise good local minumum
        - a learning rate decay can be used to reduce the learning rate
            over time in order to prevent overshooting the global minimum
            as well as start with effective learning rate

Learning Rate Decay
    - start with a large learning rate and decrease it during training
    - learning rate decay as a hyperparameter of the model
        a) decrease learning rate in reponse to the loss across epochs
        b) steadily deccrease learning rate per batch or epoch based on
            decay rate 
'''
start_learning_rate = 1
# 0.1 in practice a rather aggressive decay rate
learning_rate_decay = 0.1

'''
Demonstrating learning rate decay per step
    - update learning rate each step by reciprocal of the step count fraction
    - multiply step by decaying ratio - the further the step in training,
        the bigger the result
    - take its reciprocal (the further in training, the lower the value) and
        multiply the initial learning rate
    - added 1 ensures that the resulting algorithm never raises
        the learning rate
'''
for step in range(20):
    learning_rate = start_learning_rate * (1 / (1 + learning_rate_decay * step))
    print(learning_rate)
