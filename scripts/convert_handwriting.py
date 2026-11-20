
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/handwriting/handwriting_model.keras"

# Separate candidate: preserve the existing INT8 model.
OUTPUT_PATH = "models/handwriting/handwriting_model_int8_62class.tflite"

CALIBRATION_SAMPLES = 1000
QUICK_TEST_SAMPLES = 2000

os.makedirs("models/handwriting", exist_ok=True)


# ==========================================
# PREPROCESSING
# Must match train_handwriting.py
# ==========================================

def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.int32)

    mapped_label = tf.where(
        label < 10,
        label + 52,
        label - 10
    )

    return image, mapped_label


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading trained handwriting model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if model.input_shape != (None, 28, 28, 1):
    raise ValueError(
        f"Unexpected model input shape: {model.input_shape}"
    )

if model.output_shape != (None, 62):
    raise ValueError(
        f"Expected 62 output classes, got {model.output_shape}"
    )

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ==========================================
# PREPARE REPRESENTATIVE DATA
# ==========================================

print("\nLoading EMNIST ByClass calibration data...")

calibration_ds = tfds.load(
    "emnist/byclass",
    split="train",
    as_supervised=True
)

calibration_ds = (
    calibration_ds
    .shuffle(10000, seed=42)
    .take(CALIBRATION_SAMPLES)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
)

def representative_dataset():
    for image, label in tfds.as_numpy(calibration_ds):
        yield [np.expand_dims(image, axis=0).astype(np.float32)]


# ==========================================
# FULL INT8 CONVERSION
# ==========================================

print("\nConverting to full INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as file:
    file.write(tflite_model)

print("\nCandidate INT8 model saved:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")


# ==========================================
# VERIFY TFLITE MODEL
# ==========================================

print("\nVerifying converted model...")

interpreter = tf.lite.Interpreter(
    model_path=OUTPUT_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input shape:", input_details["shape"])
print("Input dtype:", input_details["dtype"])
print("Output shape:", output_details["shape"])
print("Output dtype:", output_details["dtype"])

if list(input_details["shape"]) != [1, 28, 28, 1]:
    raise ValueError("Incorrect INT8 input dimensions.")

if list(output_details["shape"]) != [1, 62]:
    raise ValueError("Incorrect INT8 output dimensions.")

if input_details["dtype"] != np.int8:
    raise ValueError("Input is not INT8.")

if output_details["dtype"] != np.int8:
    raise ValueError("Output is not INT8.")

input_scale, input_zero = input_details["quantization"]

if input_scale <= 0:
    raise ValueError("Invalid input quantization scale.")

print("\nQuantization verified.")


# ==========================================
# PRELIMINARY INT8 EVALUATION
# ==========================================

print("\nRunning preliminary test on EMNIST ByClass...")

test_ds = tfds.load(
    "emnist/byclass",
    split=f"test[:{QUICK_TEST_SAMPLES}]",
    as_supervised=True
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

correct = 0
total = 0

for image, label in tfds.as_numpy(test_ds):

    image = np.expand_dims(image, axis=0)

    quantized_input = np.rint(
        image / input_scale + input_zero
    )

    quantized_input = np.clip(
        quantized_input, -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details["index"]
    )

    predicted_class = int(np.argmax(prediction))
    actual_class = int(label)

    if predicted_class == actual_class:
        correct += 1

    total += 1

accuracy = correct / total * 100

print("\n========== INT8 PRELIMINARY RESULTS ==========")
print("Test samples:", total)
print(f"INT8 accuracy: {accuracy:.2f}%")
print("Output classes: 62")
print("Model size:", len(tflite_model), "bytes")
print("Candidate:", OUTPUT_PATH)
print("==============================================")

print("\nConversion and preliminary verification completed.")
