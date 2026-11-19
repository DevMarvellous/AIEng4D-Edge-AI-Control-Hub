
import os
import string
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

KERAS_PATH = "models/handwriting/handwriting_model.keras"
INT8_PATH = "models/handwriting/handwriting_model_int8.tflite"

os.makedirs("results", exist_ok=True)

print("Loading floating-point model...")
keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)

print("Loading INT8 model...")
interpreter = tf.lite.Interpreter(model_path=INT8_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]

print("Loading the full EMNIST validation split...")

test_ds = tfds.load(
    "emnist/letters",
    split="train[90%:]",
    as_supervised=True
)

def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32) - 1
    return image, label

test_ds = (
    test_ds
    .map(preprocess)
    .shuffle(25000, seed=42, reshuffle_each_iteration=False)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

true_labels = []
float_predictions = []
int8_predictions = []

print("Evaluating both models...")

for images, labels in test_ds:
    images_np = images.numpy()
    labels_np = labels.numpy()

    # Floating-point model
    float_output = keras_model(
        images,
        training=False
    ).numpy()

    float_predictions.extend(
        np.argmax(float_output, axis=1)
    )

    # INT8 model
    for image in images_np:
        quantized = np.round(
            image / input_scale + input_zero
        )

        quantized = np.clip(
            quantized, -128, 127
        ).astype(np.int8)

        quantized = np.expand_dims(quantized, axis=0)

        interpreter.set_tensor(
            input_details["index"],
            quantized
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        int8_predictions.append(
            int(np.argmax(output))
        )

    true_labels.extend(labels_np)

true_labels = np.array(true_labels)
float_predictions = np.array(float_predictions)
int8_predictions = np.array(int8_predictions)

float_accuracy = accuracy_score(
    true_labels, float_predictions
) * 100

int8_accuracy = accuracy_score(
    true_labels, int8_predictions
) * 100

agreement = np.mean(
    float_predictions == int8_predictions
) * 100

letters = list(string.ascii_uppercase)
labels_range = list(range(26))

float_report = classification_report(
    true_labels,
    float_predictions,
    labels=labels_range,
    target_names=letters,
    zero_division=0
)

int8_report = classification_report(
    true_labels,
    int8_predictions,
    labels=labels_range,
    target_names=letters,
    zero_division=0
)

report = f"""
HANDWRITING MODEL EVALUATION
========================================

Dataset: EMNIST Letters
Validation samples:{len(true_labels)}

FLOATING-POINT MODEL
Accuracy: {float_accuracy:.2f}%

INT8 QUANTIZED MODEL
Accuracy: {int8_accuracy:.2f}%

QUANTIZATION COMPARISON
Accuracy difference: {int8_accuracy - float_accuracy:+.2f} percentage points
Prediction agreement: {agreement:.2f}%

----------------------------------------
FLOAT MODEL CLASSIFICATION REPORT
----------------------------------------
{float_report}

----------------------------------------
INT8 MODEL CLASSIFICATION REPORT
----------------------------------------
{int8_report}
"""

print(report)

with open(
    "results/handwriting_evaluation.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# Save INT8 confusion matrix
cm = confusion_matrix(
    true_labels,
    int8_predictions,
    labels=labels_range
)

plt.figure(figsize=(12, 10))
plt.imshow(cm)
plt.title("INT8 Handwriting Model - Confusion Matrix")
plt.xlabel("Predicted Letter")
plt.ylabel("Actual Letter")
plt.xticks(labels_range, letters)
plt.yticks(labels_range, letters)
plt.colorbar()
plt.tight_layout()

plt.savefig(
    "results/handwriting_confusion_matrix.png",
    dpi=200
)

plt.close()

print("\nEvaluation completed.")
print("Saved results/handwriting_evaluation.txt")
print("Saved results/handwriting_confusion_matrix.png")
