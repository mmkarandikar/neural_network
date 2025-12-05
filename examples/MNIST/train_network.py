"""
Train a neural network on the normalised MNIST dataset.
We train the network to return a probability for each number between 0-9.
Thus, the output layer returns a vector of size 10.
"""

# Import modules
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
import numpy as np
import dill
from src.network.layers import DenseLayer
from src.helpers.activations import Sigmoid, Softmax, ReLU
from src.network.network_functions import train, predict

# Load training data
cutoff = 20000 # we load a subset of the full dataset
data = np.load('examples/MNIST/data/mnist_data_normalised.npz')
images = (data['training_images'].astype(np.float32))[:cutoff]
labels = (data['training_labels'].astype(int))[:cutoff]

# Define the network parameters
input_layer_size = int(images[0].size)
output_layer_size = 10
hidden_layer_size = 16

network = [
        DenseLayer(input_layer_size, hidden_layer_size),
        Softmax(),
        DenseLayer(hidden_layer_size, output_layer_size),
        Softmax(),
        ]

# Shuffle the training data to remove any patterns in the ordering of images
mapping_array = np.linspace(0, images.shape[0]-1, images.shape[0], dtype=int)
np.random.shuffle(mapping_array)
new_images = images[mapping_array]
new_labels = labels[mapping_array]

# Flatten the images to a 1D vector
images_flat = np.array([new_images[j].flatten() for j in range(new_images.shape[0])])

# Convert the labels from one number to a vector of size output_layer_size
true_labels = np.zeros((new_labels.shape[0], 10), dtype=int)
for index, value in enumerate(new_labels):
    true_labels[index, value] = 1

# Split into training, validation, and testing data
fractions = [0.7, 0.2, 0.1]
training_map = mapping_array[:int(fractions[0]*images_flat.shape[0])]
validation_map = mapping_array[int(fractions[0]*images_flat.shape[0]):int((fractions[0]+fractions[1])*images_flat.shape[0])]
testing_map = mapping_array[int((fractions[0]+fractions[1])*images_flat.shape[0]):]

training_input = images_flat[training_map, :]
validation_input = images_flat[validation_map, :]
testing_input = images_flat[testing_map, :]

training_labels = true_labels[training_map, :]
validation_labels = true_labels[validation_map, :]
testing_labels = true_labels[testing_map, :]

# Train network
network, training_loss, validation_loss = train(network, train_x=training_input, train_y=training_labels, valid_x=validation_input, valid_y=validation_labels, epochs=15, save_rate=5, learning_rate=0.005, save_path="examples/MNIST/data/saved_weights/")

# Save test data, the trained network and the losses
test_data = np.concatenate((testing_input, testing_labels), axis=1)
np.save("examples/MNIST/data/test_data", test_data)

file = open("examples/MNIST/data/trained_network", "wb")
dill.dump(network, file)
file.close()

file = open("examples/MNIST/data/losses", "wb")
dill.dump(np.array([training_loss, validation_loss]), file)
file.close()