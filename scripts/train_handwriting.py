
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
import string

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = "models/handwriting"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, "handwriting_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = (
    list(string.ascii_uppercase)
    + list(string.ascii_lowercase)
    + list(string.digits)
)

assert len(CLASS_NAMES) == 62

print("Loading EMNIST ByClass...")
print("Classes: A-Z, a-z, 0-9")
print("Total classes:", len(CLASS_NAMES))


# ==========================================
# LOAD DATASET
# ==========================================

train_ds = tfds.load(
    "emnist/byclass",
    split="train[:90%]",
    as_supervised=True
)

val_ds = tfds.load(
    "emnist/byclass",
    split="train[90%:]",
    as_supervised=True
)

test_ds = tfds.load(
    "emnist/byclass",
    split="test",
    as_supervised=True
)


# ==========================================
# PREPROCESSING AND LABEL REMAPPING
# ==========================================

def preprocess(image, label):

    # Correct EMNIST image orientation.
    image = tf.transpose(image, perm=[1, 0, 2])

    # Convert pixels to float values from 0 to 1.
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.int32)

    # Native EMNIST ByClass:
    # 0-9   = digits
    # 10-35 = uppercase letters
    # 36-61 = lowercase letters
    #
    # Our model:
    # 0-25  = uppercase letters
    # 26-51 = lowercase letters
    # 52-61 = digits

    mapped_label = tf.where(
        label < 10,
        label + 52,
        tf.where(
            label < 36,
            label - 10,
            label - 10
        )
    )

    return image, mapped_label


# ==========================================
# PREPARE DATASETS
# ==========================================

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(30000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================
# MODEL ARCHITECTURE
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        8, 3, activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        16, 3, activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        32, activation="relu"
    ),

    tf.keras.layers.Dense(
        62, activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# TRAIN
# ==========================================

print("\nStarting 62-class handwriting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12
)


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating on EMNIST ByClass test data...")

loss, accuracy = model.evaluate(test_ds)

print("\n========== TRAINING RESULTS ==========")
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("Output classes:", model.output_shape[-1])
print("======================================")


# ==========================================
# SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )

print("\nSaved Keras model:", MODEL_PATH)
print("Saved class mapping:", CLASS_NAMES_PATH)
print("\nTraining completed.")
