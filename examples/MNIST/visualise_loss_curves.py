import matplotlib.pyplot as plt
import dill
plt.style.use("plotstyle.mplstyle")

# Load losses
file = open("data/losses", "rb")
losses = dill.load(file)
file.close()

fig, ax = plt.subplots()
ax.plot(losses[0,:], c="b", label="training loss")
ax.plot(losses[1,:], c="k", ls="dashed", label="validation loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
plt.legend()
plt.show()