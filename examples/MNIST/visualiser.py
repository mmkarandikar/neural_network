import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import dill
from src.network.network_functions import predict

st.title("Neural network evaluation (MNIST dataset)")

# Load test data; path defined to be run from project root
test_data = np.load("examples/MNIST/data/test_data.npy")
test_inputs = test_data[:, :784]
test_labels = test_data[:, 784:]

# Load trained neural network
file = open("examples/MNIST/data/trained_network", "rb")
network = dill.load(file)
file.close()

# Pick random sample
if "idx" not in st.session_state:
    st.session_state.idx = 0
if st.button("Pick Random Sample"):
    st.session_state.idx = np.random.randint(0, len(test_inputs))
index = st.session_state.idx
sample_input = test_inputs[index]
sample_label = np.argmax(test_labels[index])

# Make prediction
predicted_label = np.argmax(predict(network, (sample_input).reshape(1, -1)).reshape(10))

# Display
left_col, right_col = st.columns(2)

with left_col:
    fig, ax = plt.subplots()
    ax.imshow(sample_input.reshape(28, 28), cmap="gray")
    ax.set_title(f"True: {sample_label}, Predicted: {predicted_label}")
    ax.axis("off")
    st.pyplot(fig)

with right_col:
    # Placeholder to display model architecture and performance here
    pass
