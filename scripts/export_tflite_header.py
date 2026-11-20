
from pathlib import Path
import tensorflow as tf


MODELS = [
    {
        "name": "Handwriting (62 Classes)",
        "model": Path(
            "models/handwriting/handwriting_model_int8_62class.tflite"
        ),
        "output": Path(
            "firmware/handwriting_recognition/model_data.h"
        ),
        "array": "g_handwriting_model",
        "expected_input": 1,
        "expected_output": 62,
    },
    {
        "name": "Intent (12 Classes)",
        "model": Path(
            "models/intent/intent_model_int8.tflite"
        ),
        "output": Path(
            "firmware/intent_control/intent_model_data.h"
        ),
        "array": "g_intent_model",
        "expected_input": 309,
        "expected_output": 12,
    },
]


def export_model(config):
    model = config["model"]
    output = config["output"]
    array_name = config["array"]

    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    input_size = int(inp["shape"][-1])
    output_size = int(out["shape"][-1])

    if input_size != config["expected_input"]:
        raise ValueError(
            f"{config['name']}: expected input size "
            f"{config['expected_input']}, got {input_size}"
        )

    if output_size != config["expected_output"]:
        raise ValueError(
            f"{config['name']}: expected output size "
            f"{config['expected_output']}, got {output_size}"
        )

    if inp["dtype"] != tf.int8 or out["dtype"] != tf.int8:
        raise TypeError(
            f"{config['name']}: expected INT8 input and output"
        )

    model_data = model.read_bytes()

    values = [
        ", ".join(
            f"0x{byte:02x}" for byte in model_data[i:i + 12]
        )
        for i in range(0, len(model_data), 12)
    ]

    header = f"""#pragma once
#include <cstdint>

alignas(16) const unsigned char {array_name}[] = {{
""" + ",\n".join(values) + f"""
}};

const unsigned int {array_name}_len =
    sizeof({array_name});
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header)

    print(f"\n{'=' * 50}")
    print(config["name"].upper())
    print(f"{'=' * 50}")
    print("Source model:", model)
    print("Exported header:", output)
    print("Model size:", len(model_data), "bytes")

    print("\nINPUT")
    print("Shape:", inp["shape"])
    print("Type:", inp["dtype"])
    print("Quantization:", inp["quantization"])

    print("\nOUTPUT")
    print("Shape:", out["shape"])
    print("Type:", out["dtype"])
    print("Quantization:", out["quantization"])

    print("\nREQUIRED OPERATORS")
    ops = interpreter._get_ops_details()
    for op in sorted(set(item["op_name"] for item in ops)):
        print("-", op)

    print("\nExport completed successfully.")


if __name__ == "__main__":
    for model_config in MODELS:
        export_model(model_config)

    print("\nBoth firmware model headers exported successfully.")
