
from pathlib import Path
import tensorflow as tf


MODELS = [
    {
        "name": "Handwriting",
        "model": Path("models/handwriting/handwriting_model_int8.tflite"),
        "output": Path("firmware/handwriting_recognition/model_data.h"),
        "array": "g_handwriting_model",
    },
    {
        "name": "Intent",
        "model": Path("models/intent/intent_model_int8.tflite"),
        "output": Path("firmware/intent_control/intent_model_data.h"),
        "array": "g_intent_model",
    },
]


def export_model(config):
    model = config["model"]
    output = config["output"]
    array_name = config["array"]

    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    output.parent.mkdir(parents=True, exist_ok=True)

    model_data = model.read_bytes()

    values = [
        ", ".join(f"0x{byte:02x}" for byte in model_data[i:i + 12])
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

    output.write_text(header)

    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"\n{'=' * 45}")
    print(f"{config['name'].upper()} MODEL")
    print(f"{'=' * 45}")
    print("Header:", output)
    print("Model bytes:", len(model_data))

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


if __name__ == "__main__":
    for model_config in MODELS:
        export_model(model_config)

    print("\nBoth model headers exported successfully.")
