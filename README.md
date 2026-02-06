# Neural network in python without tensorflow/keras/torch
In this package, we build a neural network using only the standard python libraries. The objective is to learn the principles of designing neural networks -- loss functions, activations, backpropagation, and so on.

* The current version has the following:
    + Functioning implementation of dense layers
    + Backpropagation via regular gradient descent
    + Choice of activations between ReLU, Sigmoid and Softmax
    + An example use case with the MNIST dataset of handwritten digits
    + Implemented batches

* Here is a list of features yet to be added:
    + Implement normalisation between layers (or add BatchNorm)
    + Implement convolutional layer backprop
    + Add transformer layer
    + Add flavours of gradient descent