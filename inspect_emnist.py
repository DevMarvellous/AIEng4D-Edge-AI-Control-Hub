
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

print("Loading EMNIST Letters...")

dataset = tfds.load(
    "emnist/letters",
    split="train",
    as_supervised=True
)

samples = {}

for image, label in tfds.as_numpy(dataset):
    label = int(label)

    if label not in samples:
        samples[label] = image.squeeze()

    if len(samples) == 26:
        break

fig, axes = plt.subplots(4, 7, figsize=(12, 8))
axes = axes.flatten()

for i in range(26):
    label = i + 1
    image = samples[label]

    # Correct EMNIST orientation
    image = image.T

    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(chr(64 + label))
    axes[i].axis("off")

for i in range(26, len(axes)):
    axes[i].axis("off")

plt.suptitle("EMNIST Letters — Orientation Check")
plt.tight_layout()
plt.savefig("emnist_orientation_check.png")
plt.show()

print("Saved: emnist_orientation_check.png")
