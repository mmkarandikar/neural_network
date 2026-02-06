"""
This module defines all layer classes.
"""

# Import modules
import numpy as np
import scipy as sp


# Implement base class
class Layer:
    """
    This class defines the base layer class. All specific layer types will inherit from this class.
    """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(
            2 / self.input_size
        )
        self.biases = np.zeros(self.output_size)

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
        self.inputs = inputs
        self.output = self.inputs @ self.weights + self.biases
        return self.output

    def backward_pass(self, output_gradient, learning_rate):
        batch_size = self.inputs.shape[0]
        input_gradient = output_gradient @ self.weights.T
        self.weights_gradient = self.inputs.T @ output_gradient / batch_size
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * np.mean(output_gradient, axis=0)
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

    def backward_pass(self, output_gradient):
        return output_gradient * self.activation_derivative(self.inputs)


class ConvLayer:
    """Defines a convolutional layer"""

    def __init__(self, filter_shape, padding, stride, n_filters=1, n_channels=1):
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.weights = np.random.randn(
            self.n_filters, self.n_channels, *self.filter_shape
        ) * np.sqrt(2 / np.prod(self.filter_shape))
        self.biases = np.zeros(n_filters)

    def forward_pass(self, inputs):
        self.inputs = inputs

        # Pad the last two axes of the input tensor
        padded_inputs = np.pad(
            self.inputs,
            pad_width=(
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
        )

        # Extract all patches from the input, slice according to the stride
        patch_matrix = np.lib.stride_tricks.sliding_window_view(
            padded_inputs, self.filter_shape, axis=(1, 2)
        )[:, :: self.stride, :: self.stride, :, :, :]

        # Reshape before the dot product; the -1 lets reshape infer the shape as the product of the remaining axes
        patch_matrix = patch_matrix.reshape(*patch_matrix.shape[:-3], -1)
        weights_flat = self.weights.reshape(-1, np.prod(self.weights.shape[1:]))
        self.output = patch_matrix @ weights_flat.T + self.biases
        return self.output

    def pooling_forward(self, pooling_filter_shape, pooling_stride):
        pooling_patches = np.lib.stride_tricks.sliding_window_view(
            self.output, pooling_filter_shape, axis=(1, 2)
        )[:, ::pooling_stride, ::pooling_stride, :, :]

        # Max pooling - take the maximum of the reshaped patches
        self.output = np.max(
            pooling_patches.reshape(*pooling_patches.shape[:-2], -1), -1
        )
        return self.output

    def backward_pass(self, output_gradient, learning_rate):
        # Calculate gradients
        print(output_gradient.shape, self.weights.shape)
        # input_gradient = output_gradient @ self.weights.T
        # self.weights_gradient = np.dot(output_gradient, self.inputs.T)
        # Update weights
        # Update biases
        self.bias_gradient = np.mean(output_gradient, axis=(0, 1, 2))
        return self.bias_gradient


class Flatten:
    """
    Flattens the output of a convolutional layer in preparation for a dense layer input
    """

    def __init__(self):
        pass

    def forward_pass(self, inputs):
        self.input_shape = inputs.shape
        inputs_flat = inputs.reshape(self.input_shape[0], -1)
        return inputs_flat

    def backward_pass(self, output_gradient):
        return output_gradient.reshape(self.input_shape)
