"""
This module defines all layer classes.
"""

# Import modules
import numpy as np

# Implement base class
class Layer:
    """
    This class defines the base layer class. All specific layer types will inherit from this class.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size, self.output_size)  
        self.biases = np.random.rand(self.output_size, 1)
    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

# Implement input layer class
class InputLayer(Layer):
    def ___init__(self, input_data):
        super().__init__(input_data)
        self.size = input_data.size

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

# Implement dense layer class
class DenseLayer(Layer):
    """
    This class defines a dense/fully-connected layer that inherits from the base layer class.
    """
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward_pass(self, inputs):
        self.input = inputs
        self.output = np.dot(self.weights.T, self.input) + self.biases
        return self.output

    def backward_pass(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights, output_gradient)
        self.weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * self.weights_gradient.T
        self.biases -= learning_rate * output_gradient
        return input_gradient


# Implement activation layer
class Activation(Layer):
    """Defines the activation layer"""
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_pass(self, inputs):
        self.inputs = inputs
        return self.activation(inputs)
    
    def backward_pass(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.inputs))
    