"""
Use the trained network to make predictions
"""

# Import modules
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)
import numpy as np
import dill
from src.network.network_functions import predict


def calculate_accuracy(file_name: str) -> np.array:
    """
    Load a trained network and calculate its classification accuracy
    """
    # Load trained network
    file = open(f"{file_name}", "rb")
    network = dill.load(file)
    file.close()

    # Make predictions
    predictions = predict(network, test_inputs).argmax(axis=1)
    truth = test_labels.argmax(axis=1)

    accuracy = (sum((predictions == truth).astype(int)) / test_inputs.shape[0]) * 100
    return np.round(accuracy, 3)


# Load test data
test_data = np.load("examples/MNIST/data/test_data.npy")
test_inputs = test_data[:, :784]
test_labels = test_data[:, 784:]


save_path = "examples/MNIST/data/saved_weights/"
for file in sorted(os.listdir(save_path)):
    accuracy = calculate_accuracy(save_path + file)
    print(f"Epoch: {int(file[-3:])}, Accuracy: {accuracy}%")
