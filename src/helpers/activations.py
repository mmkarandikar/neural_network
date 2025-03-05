"""
This modules defines a library of activation functions, inheriting from the Activation class.
"""

import numpy as np
from . import Activation

class ReLU(Activation):
    """The ReLU function"""
    def __init__(self):
        def act(inputs):
            return np.maximum(np.zeros_like(inputs), inputs)
        
        def act_der(inputs):
            return act(inputs) / inputs
        super().__init__(act, act_der)


class Softmax(Activation):
    """Softmax activation function, returns an output between 0 and 1."""
    def __init__(self):
        def act(inputs):
            return np.exp(inputs) / np.sum(np.exp(inputs))
               
        def act_der(inputs):
            return act(inputs) * (1 - act(inputs))
        super().__init__(act, act_der)


class Sigmoid(Activation):
    """The logistic function"""
    def __init__(self):
        def act(inputs):
            return 1 / (1 + np.exp(-inputs))
        
        def act_der(inputs):
            return np.multiply(act(inputs), (1 - act(inputs)))
        super().__init__(act, act_der)
