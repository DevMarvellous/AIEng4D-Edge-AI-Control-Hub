
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

os.makedirs("models/handwriting", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading EMNIST Letters...")

train_ds = tfds.load(
    "emnist/letters",
    split="train[:90%]",
    as_supervised=True
)

val_ds = tfds.load(
    "emnist/letters",
    split="train[90%:]",
    as_supervised=True
)

test_ds = tfds.load(
    "emnist/letters",
    split="test",
    as_supervised=True
)

def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32) - 1
    return image, label

train_ds = train_ds.map(preprocess).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(26, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining handwriting model...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

print("\nEvaluating on test data...")
loss, accuracy = model.evaluate(test_ds)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

model.save("models/handwriting/handwriting_model.keras")

print("\nSaved handwriting model successfully.")
