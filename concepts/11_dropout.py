import numpy as np
import random

'''
Dropout in forward pass
    - dropout layer disables some neurons, while others pass through unchanged
    - prevent a neural network from becoming too dependent on any neuron
        or for any neuron to be relied upon entirely in a specific instance
        - reliance can be common if a model overfits the data
        - also helps with co-adoption, which happens when neurons depend on the
            output values of other neurons and do not learn the underlying
            function on their own
        - furthermore, can help with noise and other perturbations in the data
    - randomly disables neurons at a given rate during every forward pass
    => forces the network to learn how to make accurate predictions with
        only a random part of neurons remaining
    => forces the model to use more neurons for the same purpose, resulting in
        a higher chance of learning underlying function that describes the data
    - implemented by setting neuron outputs to 0 with a certain probability
'''

# Hyperparameter for percentage of neurons to disable in that layer
dropout_rate = 0.5

# Example output of 10 values
example_output = [0.27, -1.03, 0.67, 0.99, 0.05,
                  -0.37, -2.01, 1.13, -0.07, 0.73]

# Repeat until dropout_rate is reached
while True:
    # Randomly choose index and set value to 0
    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0

    # Check if dropout_rate is reached, as accidenlty the same indexes can be
    # chosen multiple times
    dropped_out = 0
    for value in example_output:
        if value == 0:
            dropped_out += 1
    
    # If dropout_rate is reached, break loop
    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)

'''
Implementation using Binomial distribution
    - cleaner implementation can be reached using np.random.binomial
    - np.random.binomial(n, p, size), where:
        - n: number of concurrent trials, i.e. trials per experiment
        - p: probability of the true value of the trial
        - size: number of experiemtns to run
        - returns: array of size size with the results of the successfull trials
            per experiment
    - np.random.binomial is based on probailities, so there will be times when 
        the results will be as above, sometimes no neurons zero out, or all
    - on average, these random draws will tend towards the probability desired
'''

# Simple example
print(np.random.binomial(2, 0.5, size=10))

# Hyperparameter for percentage of neurons to disable in that layer
dropout_rate = 0.3

# Example output of 10 values
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])

example_output *= np.random.binomial(1, 1 - dropout_rate, example_output.shape)

print(example_output)

'''
Dropout in training vs. testing
    - dropout should not be utilized when predicting
    - simply ommitting a previous dropout during prediction in testing does not
        work, as magnitude of inputs to the next neurons can be dramatically
        different
    - e.g. when during training 50% of neurons are disabled, and then the model
        used all neurons to predict, statistically the output could be twice
        as big (or small, depending on operation)
    - to prevent, in training the data is scaled up after a dropout during the
        training phase, to mimic the mean of the sum when all neurons output
        their values
'''

# Hyperparameter for percentage of neurons to disable in that layer
dropout_rate = 0.2

# Example output of 10 values
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])
print(f'sum initial {sum(example_output)}')

sums = []
for i in range(100000):
    example_output_Two = example_output * \
        np.random.binomial(1, 1 - dropout_rate, example_output.shape) / \
        (1-dropout_rate)
    sums.append(sum(example_output_Two))

print(f'mean sum: {np.mean(sums)}')

'''
Dropout in backward pass
    - as before, partial derivative of the dropout operation must be calculated
    - see README.md for mathematical details on dropout derivative
'''