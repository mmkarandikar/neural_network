"""
This module defines useful functions for training the network.
"""

import os
import time
import dill
import shutil
import numpy as np
from tqdm import tqdm
from src.helpers.loss_functions import MeanSquaredError, CategoricalCrossEntropyLoss
from src.helpers.activations import Softmax, ReLU, LeakyReLU
from src.network.layers import DenseLayer


def train(
    network: list,
    train_x: np.array,
    train_y: np.array,
    loss_function: callable,
    valid_x: np.array = None,
    valid_y: np.array = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    save_rate: int = None,
    save_path: str = None,
) -> tuple:
    """
    This function trains the input network with the training data.

    Args:
        network (list): The list of network layers
        train_x (np.array): Network training input
        train_y (np.array): True output of the training samples
        valid_x (np.array): Network validation input
        valid_y (np.array): True output of the validation samples
        epochs (int): Number of epochs
        batch_size (int): Size of batches for training data
        learning_rate (float): Learning rate
        loss_function (callable): The chosen loss function

    Returns:
        trained_network (list): List of network layers with updated weights and biases
        training_loss (np.array): Training loss at each epoch
        validation_loss (np.array): Validation loss at each epoch
    """

    # Define empty arrays for recording loss
    training_loss = np.empty(epochs)
    validation_loss = np.empty(epochs)
    t0 = time.time()

    reversed_network = list(reversed(network))  # For backpropagation

    if loss_function is CategoricalCrossEntropyLoss:
        assert type(reversed_network[0]) == type(
            Softmax()
        ), "When using CategoricalCrossEntropyLoss, last activation layer must be Softmax"
        reversed_network = reversed_network[1:]

    # Create directory to save weights; remove old weights
    if save_rate is not None:
        if os.path.exists(save_path):
            for filename in os.listdir(save_path):
                file = save_path + filename
                try:
                    os.remove(file)
                except OSError as e:
                    pass
        else:
            os.mkdirs(save_path)

    # Loop over epochs
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_valid_loss = 0

        for batch in range(0, train_x.shape[0], batch_size):
            batch_x = train_x[batch : batch + batch_size]
            batch_y = train_y[batch : batch + batch_size]

            # Forward pass
            layer_input = batch_x
            for layer in network:
                layer_output = layer.forward_pass(inputs=layer_input)
                layer_input = layer_output
            predicted_output = layer_output

            # Loss evaluation
            loss_instance = loss_function(predicted_output, batch_y)
            epoch_train_loss += loss_instance.calculate_loss()
            loss_grad = loss_instance.calculate_loss_gradient()

            # Backward pass
            backward_layer_in = loss_grad
            for layer in reversed_network:
                if type(layer).__name__ == "DenseLayer":
                    backward_layer_out = layer.backward_pass(
                        backward_layer_in, learning_rate
                    )
                else:
                    backward_layer_out = layer.backward_pass(backward_layer_in)
                backward_layer_in = backward_layer_out

        epoch_train_loss /= train_x.shape[0]
        training_loss[epoch] = epoch_train_loss

        # Validation
        prediction = predict(network, valid_x)
        epoch_valid_loss = loss_function(prediction, valid_y).calculate_loss()

        epoch_valid_loss /= valid_x.shape[0]
        validation_loss[epoch] = epoch_valid_loss

        # Save weights
        if save_rate is not None:
            if (epoch % save_rate == 0) or (epoch == epochs - 1):
                file = open(f"{save_path}/epoch_{epoch:03d}", "wb")
                dill.dump(network, file)
                file.close()
            else:
                pass
        else:
            pass

        # Display losses
        print(
            f"Epoch {epoch+1}/{epochs}; train loss: {np.round(epoch_train_loss, 3)}, valid loss: {np.round(epoch_valid_loss, 3)}"
        )
    t1 = time.time()
    print(f"Training finished; took {np.round(t1-t0, 3)} seconds.")
    return network, training_loss, validation_loss


def predict(network: list, inputs: np.array) -> np.array:
    """
    This function predicts the output of a trained network.

    Args:
        network (list): The trained network, provided as a list of its layers
        inputs (np.array): N-dimensional input array such that inputs[i] returns the i-th input
    Returns:
        network_prediction (np.array): Prediction of the network
    """

    layer_in = inputs
    for layer in network:
        layer_out = layer.forward_pass(inputs=layer_in)
        layer_in = layer_out

    return layer_out
