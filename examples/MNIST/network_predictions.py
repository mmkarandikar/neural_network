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
    predictions = predict(network, test_inputs)[:, :, 0].argmax(axis=1)
    truth = test_labels.argmax(axis=1)

    accuracy = sum((predictions == truth).astype(int)) / test_inputs.shape[0]
    return accuracy

# Load test data
test_data = np.load("examples/MNIST/data/test_data.npy")
test_inputs = test_data[:, :784]
test_labels = test_data[:, 784:]

for i in range(0, 21, 5):
    file_name = f"examples/MNIST/data/saved_weights/epoch_{i:03d}"
    accuracy = calculate_accuracy(file_name)
    print(f"Accuracy: {accuracy*100}%")
