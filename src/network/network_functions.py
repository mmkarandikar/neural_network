"""
This module defines useful functions for training the network.
"""
import numpy as np
import time
import dill
from tqdm import tqdm
from . import MeanSquaredError

def train(network: list, train_x: np.array, train_y: np.array, valid_x: np.array = None,
            valid_y: np.array = None, epochs: int = 100, learning_rate: float = 0.01,
                 save_rate: int = None) -> tuple:
    """
    This function trains the input network with the training data.
    
    Args:
        network (list): The list of network layers
        train_x (np.array): Network training input
        train_y (np.array): True output of the training samples
        valid_x (np.array): Network validation input
        valid_y (np.array): True output of the validation samples
        epochs (int): Number of epochs
        learning_rate (float): Learning rate

    Returns:
        trained_network (list): List of network layers with updated weights and biases
        training_loss (np.array): Training loss at each epoch
        validation_loss (np.array): Validation loss at each epoch
    """

    # Define empty arrays for recording loss
    training_loss = np.empty(epochs)
    validation_loss = np.empty(epochs)
    t0 = time.time()

    for epoch in range(epochs):
        epoch_train_loss, epoch_valid_loss = 0.0, 0.0
        for sample_x, sample_y in zip(train_x, train_y):

            # Reshape training data
            sample_x = sample_x.reshape(sample_x.size, 1)
            sample_y = sample_y.reshape(sample_y.size, 1)

            # Forward pass
            layer_input = sample_x
            for layer in network:
                layer_output = layer.forward_pass(inputs=layer_input)
                layer_input = layer_output    
            predicted_output = layer_output

            # Loss eval
            loss_function = MeanSquaredError(predicted_output, sample_y)
            loss = loss_function.calculate_loss()
            loss_grad = loss_function.calculate_loss_gradient()
            epoch_train_loss += loss

            # Backward pass
            backward_layer_in = loss_grad
            for layer in list(reversed(network)):
                backward_layer_out = layer.backward_pass(backward_layer_in, learning_rate)
                backward_layer_in = backward_layer_out

        epoch_train_loss /= train_x.shape[0]
        training_loss[epoch] = epoch_train_loss

        # Validation loss
        for sample_x, sample_y in zip(valid_x, valid_y):
            # prediction = predict(network, sample_x.reshape(sample_x.size, 1))

            prediction = predict(network, sample_x)
            truth = sample_y.reshape(sample_y.size, 1)

            loss = MeanSquaredError(prediction, truth).calculate_loss()
            epoch_valid_loss += loss

        epoch_valid_loss /= valid_x.shape[0]
        validation_loss[epoch] = epoch_valid_loss

        if save_rate is not None:
            if epoch%save_rate == 0:
                file = open(f"data/saved_weights/epoch_{epoch:03d}", "wb")
                dill.dump(network, file)
                file.close()
            else:
                pass
        else:
            pass
        print(f"Epoch {epoch+1}/{epochs}; train loss: {np.round(epoch_train_loss, 3)}, val loss: {np.round(epoch_valid_loss, 3)}")
    t1 = time.time()
    print(f"Training finished; took {np.round(t1-t0, 3)} seconds.")
    return network, training_loss, validation_loss


def predict(network: list, inputs: np.array) -> np.array:
    """
    This function predicts the output of a trained network.
    
    Args:
        network (list): The trained network, provided as a list of its layers
        input (np.array): The input for the network. If it is a 2d array, it should be structured
            such that input[i] returns the i-th input
    Returns:
        network_prediction (np.array): Prediction of the network
    """
    try:
        input_shape = inputs.shape[1]
        network_output = []
        for example in inputs:
            layer_input = example.reshape(example.size, 1)
            for layer in network:
                layer_output = layer.forward_pass(inputs=layer_input)
                layer_input = layer_output
            network_output.append(layer_output)
        network_prediction = np.array(network_output)
        return network_prediction

    except IndexError:
        layer_input = inputs.reshape(inputs.size, 1)
        for layer in network:
            layer_output = layer.forward_pass(inputs=layer_input)
            layer_input = layer_output
        return layer_output
