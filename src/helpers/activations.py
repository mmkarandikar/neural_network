"""
This modules defines a library of activation functions, inheriting from the Activation class.
"""

import numpy as np
from . import Activation


class ReLU(Activation):
    """The ReLU function"""

    def __init__(self):
        def act(inputs):
            return np.maximum(0, inputs)

        def act_der(inputs):
            return (inputs > 0).astype(inputs.dtype)

        super().__init__(act, act_der)


class LeakyReLU(Activation):
    """
    ReLU that allows small negative values to prevent dead neurons
    """

    def __init__(self, alpha=0.01):
        def act(inputs):
            return np.where(inputs > 0, inputs, alpha * inputs)

        def act_der(inputs):
            return np.where(inputs > 0, 1, alpha)

        super().__init__(act, act_der)


class Softmax(Activation):
    """Softmax activation function, returns an output between 0 and 1.
    Only use in combination with CategoricalCrossEntropyLoss.
    """

    def __init__(self):
        def act(inputs):
            shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
            exp_shifted = np.exp(shifted_inputs)
            return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # For CategoricalCrossEntropyLoss, we skip the derivative calculation
        act_der = None
        super().__init__(act, act_der)


class Sigmoid(Activation):
    """The logistic function"""

    def __init__(self):
        def act(inputs):
            return 1 / (1 + np.exp(-inputs))

        def act_der(inputs):
            return np.multiply(act(inputs), (1 - act(inputs)))

        super().__init__(act, act_der)
