# Neural network in python without tensorflow/keras/torch
In this package, we build a neural network using only the standard python libraries. The objective is to learn the principles of designing neural networks -- loss functions, activations, backpropagation, and so on.

The current version has the following:
    Functioning implementation of dense layers
    Backpropagation via regular gradient descent
    Choice of activations between ReLU, Sigmoid and Softmax
    An example use case with the MNIST dataset of handwritten digits

Here is a list of features yet to be added:
    Implement batches
    Implement normalisation between layers (or add BatchNorm)
    Add more loss functions
    Implement layers (recurrent, convolutionl)
    Add flavours of gradient descent
    Implement a network class