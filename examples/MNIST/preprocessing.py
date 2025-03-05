"""
Normalise MNIST images to zero mean and unit variance.
"""

# Import modules
import numpy as np

# Load data
mnist_data = np.load("data/mnist.npz")
images = mnist_data["x_train"].astype(np.float32)
labels = mnist_data["y_train"].astype(np.float32)

# Normalise and save
img_mean = np.mean(images, axis=(1,2))[:, np.newaxis, np.newaxis]
img_std = np.std(images, axis=(1,2))[:, np.newaxis, np.newaxis]
img_normalised = (mages - img_mean) / img_std
np.savez("data/mnist_data_normalised", training_images=img_normalised, training_labels=labels)
