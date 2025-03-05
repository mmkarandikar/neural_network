"""
This is a library of loss functions for the neural network.
"""
# Import modules
import numpy as np

# Implement loss functions as classes
class MeanSquaredError:
    def __init__(self, output, truth):
        self.output = output
        self.truth = truth

    def calculate_loss(self):
        self.loss = 0.5 * np.sum((self.output - self.truth)**2)
        return self.loss

    def calculate_loss_gradient(self):
        self.loss_derivative = (self.output - self.truth)
        return self.loss_derivative