
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

# ==========================================
# CONFIGURATION
# ==========================================

KERAS_PATH = "models/handwriting/handwriting_model.keras"

INT8_PATH = (
    "models/handwriting/"
    "handwriting_model_int8_62class.tflite"
)

EVALUATION_SAMPLES = 10000
RANDOM_SEED = 42

REPORT_PATH = (
    "results/handwriting_evaluation_62class.txt"
)

MATRIX_PATH = (
    "results/handwriting_confusion_matrix_62class.png"
)

os.makedirs("results", exist_ok=True)

# Class order:
# A-Z, a-z, 0-9

CLASS_NAMES = (
    list(string.ascii_uppercase)
    + list(string.ascii_lowercase)
    + list(string.digits)
)

LABELS = list(range(62))


# ==========================================
# PREPROCESSING
# Must match train_handwriting.py
# ==========================================

def preprocess(image, label):

    image = tf.transpose(
        image,
        perm=[1, 0, 2]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    label = tf.cast(
        label,
        tf.int32
    )

    mapped_label = tf.where(
        label < 10,
        label + 52,
        label - 10
    )

    return image, mapped_label


# ==========================================
# CHECK MODEL FILES
# ==========================================

print("=" * 55)
print("62-CLASS HANDWRITING MODEL EVALUATION")
print("=" * 55)

if not os.path.exists(KERAS_PATH):
    raise FileNotFoundError(KERAS_PATH)

if not os.path.exists(INT8_PATH):
    raise FileNotFoundError(
        f"Candidate INT8 model not found: {INT8_PATH}"
    )


# ==========================================
# LOAD KERAS MODEL
# ==========================================

print("\nLoading floating-point model...")

keras_model = tf.keras.models.load_model(
    KERAS_PATH,
    compile=False
)

if keras_model.output_shape[-1] != 62:
    raise ValueError(
        f"Expected 62 Keras outputs, "
        f"got {keras_model.output_shape}"
    )

print("Keras input:", keras_model.input_shape)
print("Keras output:", keras_model.output_shape)


# ==========================================
# LOAD INT8 MODEL
# ==========================================

print("\nLoading candidate INT8 model...")

interpreter = tf.lite.Interpreter(
    model_path=INT8_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = (
    input_details["quantization"]
)

if list(input_details["shape"]) != [1, 28, 28, 1]:
    raise ValueError(
        f"Unexpected INT8 input shape: "
        f"{input_details['shape']}"
    )

if list(output_details["shape"]) != [1, 62]:
    raise ValueError(
        f"Unexpected INT8 output shape: "
        f"{output_details['shape']}"
    )

if input_details["dtype"] != np.int8:
    raise ValueError("INT8 input dtype check failed.")

if output_details["dtype"] != np.int8:
    raise ValueError("INT8 output dtype check failed.")

if input_scale <= 0:
    raise ValueError("Invalid input quantization scale.")

print("INT8 input:", input_details["shape"])
print("INT8 output:", output_details["shape"])
print("Input quantization scale:", input_scale)


# ==========================================
# LOAD TEST DATA
# ==========================================

print(
    f"\nLoading {EVALUATION_SAMPLES:,} "
    "EMNIST ByClass test samples..."
)

test_ds = tfds.load(
    "emnist/byclass",
    split="test",
    as_supervised=True
)

test_ds = (
    test_ds
    .shuffle(
        20000,
        seed=RANDOM_SEED,
        reshuffle_each_iteration=False
    )
    .take(EVALUATION_SAMPLES)
    .map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================
# EVALUATE BOTH MODELS
# ==========================================

true_labels = []
float_predictions = []
int8_predictions = []

print("\nEvaluating both models...")

for batch_number, (images, labels) in enumerate(test_ds):

    images_np = images.numpy()
    labels_np = labels.numpy()

    # --------------------------------------
    # FLOATING-POINT MODEL
    # --------------------------------------

    float_output = keras_model(
        images,
        training=False
    ).numpy()

    batch_float_predictions = np.argmax(
        float_output,
        axis=1
    )

    float_predictions.extend(
        batch_float_predictions
    )

    # --------------------------------------
    # INT8 MODEL
    # --------------------------------------

    for image in images_np:

        image = np.expand_dims(
            image,
            axis=0
        )

        quantized_input = np.rint(
            image / input_scale + input_zero
        )

        quantized_input = np.clip(
            quantized_input,
            -128,
            127
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            quantized_input
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        predicted_class = int(
            np.argmax(output)
        )

        int8_predictions.append(
            predicted_class
        )

    true_labels.extend(labels_np)

    if (batch_number + 1) % 10 == 0:
        print(
            f"Processed "
            f"{len(true_labels):,} samples..."
        )


# ==========================================
# CALCULATE METRICS
# ==========================================

true_labels = np.asarray(true_labels)
float_predictions = np.asarray(float_predictions)
int8_predictions = np.asarray(int8_predictions)

if len(true_labels) == 0:
    raise RuntimeError("No evaluation samples were processed.")

float_accuracy = accuracy_score(
    true_labels,
    float_predictions
) * 100

int8_accuracy = accuracy_score(
    true_labels,
    int8_predictions
) * 100

agreement = np.mean(
    float_predictions == int8_predictions
) * 100

accuracy_difference = (
    int8_accuracy - float_accuracy
)


# ==========================================
# CLASSIFICATION REPORTS
# ==========================================

float_report = classification_report(
    true_labels,
    float_predictions,
    labels=LABELS,
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4
)

int8_report = classification_report(
    true_labels,
    int8_predictions,
    labels=LABELS,
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4
)


# ==========================================
# GENERATE FINAL REPORT
# ==========================================

report = f"""
62-CLASS HANDWRITING MODEL EVALUATION
=======================================================

DATASET
-------------------------------------------------------
Dataset: EMNIST ByClass
Evaluation samples: {len(true_labels):,}
Output classes: 62
Class order: A-Z, a-z, 0-9

FLOATING-POINT MODEL
-------------------------------------------------------
Accuracy: {float_accuracy:.2f}%

INT8 QUANTIZED MODEL
-------------------------------------------------------
Accuracy: {int8_accuracy:.2f}%

QUANTIZATION COMPARISON
-------------------------------------------------------
Accuracy difference: {accuracy_difference:+.2f} percentage points
Prediction agreement: {agreement:.2f}%

FLOATING-POINT CLASSIFICATION REPORT
-------------------------------------------------------
{float_report}

INT8 CLASSIFICATION REPORT
-------------------------------------------------------
{int8_report}

=======================================================
"""

print(report)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ==========================================
# INT8 CONFUSION MATRIX
# ==========================================

print("Generating confusion matrix...")

cm = confusion_matrix(
    true_labels,
    int8_predictions,
    labels=LABELS
)

plt.figure(figsize=(20, 18))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "62-Class INT8 Handwriting Confusion Matrix"
)

plt.xlabel("Predicted Character")
plt.ylabel("Actual Character")

plt.xticks(
    LABELS,
    CLASS_NAMES,
    rotation=90,
    fontsize=7
)

plt.yticks(
    LABELS,
    CLASS_NAMES,
    fontsize=7
)

plt.colorbar()
plt.tight_layout()

plt.savefig(
    MATRIX_PATH,
    dpi=200
)

plt.close()


# ==========================================
# COMPLETION
# ==========================================

print("\nEVALUATION COMPLETED")
print("----------------------------------------")
print(f"Samples evaluated: {len(true_labels):,}")
print(f"Float accuracy: {float_accuracy:.2f}%")
print(f"INT8 accuracy: {int8_accuracy:.2f}%")
print(f"Model agreement: {agreement:.2f}%")
print("----------------------------------------")
print("Report:", REPORT_PATH)
print("Confusion matrix:", MATRIX_PATH)
