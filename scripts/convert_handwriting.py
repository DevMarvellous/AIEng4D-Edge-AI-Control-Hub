
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_PATH = "models/handwriting/handwriting_model.keras"
OUTPUT_PATH = "models/handwriting/handwriting_model_int8.tflite"

os.makedirs("models/handwriting", exist_ok=True)

print("Loading trained handwriting model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Preparing EMNIST calibration data...")

def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32) - 1
    return image, label

calibration_ds = tfds.load(
    "emnist/letters",
    split="train[:500]",
    as_supervised=True
)

calibration_ds = calibration_ds.map(preprocess)

def representative_dataset():
    for image, label in calibration_ds:
        yield [np.expand_dims(image.numpy(), axis=0)]

print("Converting model to full INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("\nINT8 model saved:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")

print("\nTesting converted model...")

interpreter = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input:", input_details["shape"])
print("Input dtype:", input_details["dtype"])
print("Output:", output_details["shape"])
print("Output dtype:", output_details["dtype"])

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

test_ds = tfds.load(
    "emnist/letters",
    split="test",
    as_supervised=True
)

test_ds = test_ds.map(preprocess).take(2000)

correct = 0
total = 0

for image, label in tfds.as_numpy(test_ds):

    image = np.expand_dims(image, axis=0)

    quantized_input = np.round(
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

print("\n========== INT8 EVALUATION ==========")
print("Test samples:", total)
print(f"INT8 accuracy: {accuracy:.2f}%")
print("Model size:", len(tflite_model), "bytes")
print("=====================================")
