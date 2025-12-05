import numpy as np
import matplotlib.pyplot as plt

mnist_data = np.load("examples/MNIST/data/mnist.npz")
training_images = mnist_data["x_train"].astype(np.float32)
training_labels = mnist_data["y_train"].astype(np.float32)

j = np.random.randint(0, training_images.shape[0])
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(training_images[j], cmap="gray")
plt.show()