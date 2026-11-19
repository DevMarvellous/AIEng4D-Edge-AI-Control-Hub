
"""
AIEng4D - Intent Classification and Command Validation V3

Run from the project root:
    python train_intent_v3.py

Required source dataset:
    train_intent_esp32.py

The source file is parsed with ast.literal_eval. It is NOT executed.

Outputs:
    models/intent/intent_model_v3.keras
    models/intent/intent_model_v3_int8.tflite
    models/intent/vocabulary_v3.json
    models/intent/classes_v3.json
    models/intent/metadata_v3.json
    models/intent/command_parser_config.json
    results/intent_v3_results.txt

Important:
- Neural network classifies the requested hardware action.
- Deterministic parsing extracts parameters.
- Domain checks and parameter validation reject unsupported commands.
- This script does not control hardware.
"""

import ast
import json
import os
import re
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

SOURCE_SCRIPT = Path("train_intent_esp32.py")

OUTPUT_DIR = Path("models/intent")
RESULTS_DIR = Path("results")

MODEL_NAME = "AIEng4D Intent Model V3"

# Initial rejection settings.
# These are starting values, NOT scientifically established limits.
MIN_CONFIDENCE = 0.75
MIN_MARGIN = 0.15
MIN_TOKEN_COVERAGE = 0.35

# Initial software validation limit.
# The actual safe mechanical range MUST be checked on the real motor.
MOTOR_MIN_DEGREES = 0
MOTOR_MAX_DEGREES = 360

UNKNOWN_CLASS = "UNKNOWN"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE EXISTING DATASET WITHOUT EXECUTING OLD SCRIPT
# ============================================================

def load_literal_dictionary(script_path, variable_name):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Cannot find {script_path}.\n"
            "Keep your original training script in this folder, "
            "or change SOURCE_SCRIPT at the top of this file."
        )

    tree = ast.parse(
        script_path.read_text(encoding="utf-8"),
        filename=str(script_path)
    )

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == variable_name:
                        value = ast.literal_eval(node.value)
                        if not isinstance(value, dict):
                            raise TypeError(
                                f"{variable_name} must be a dictionary."
                            )
                        return value

    raise ValueError(
        f"Could not find a literal dictionary named "
        f"{variable_name} in {script_path}."
    )


training_data = load_literal_dictionary(
    SOURCE_SCRIPT, "training_data"
)

test_data = load_literal_dictionary(
    SOURCE_SCRIPT, "test_data"
)

REQUIRED_CLASSES = [
    "LED_ON",
    "LED_OFF",
    "RGB_SET",
    "MATRIX_SHOW",
    "MATRIX_CLEAR",
    "MOTOR_MOVE",
    "MOTOR_CW",
    "MOTOR_CCW",
    "MOTOR_STOP",
]

if list(training_data.keys()) != REQUIRED_CLASSES:
    missing = set(REQUIRED_CLASSES) - set(training_data.keys())
    extra = set(training_data.keys()) - set(REQUIRED_CLASSES)

    raise ValueError(
        f"Unexpected source classes. Missing={missing}, extra={extra}"
    )

for intent in REQUIRED_CLASSES:
    if intent not in test_data:
        raise ValueError(f"Missing test data for {intent}")


# ============================================================
# 3. UNKNOWN TRAINING DATA
# ============================================================

# These examples teach the model that ordinary unrelated language
# is not one of the supported hardware actions.
#
# The domain and parameter checks later provide additional rejection.
# No finite UNKNOWN dataset can cover every possible sentence.

unknown_training = [
    "my name is martel",
    "my name is marvel",
    "what is your name",
    "how are you doing today",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you very much",
    "tell me a joke",
    "tell me a story",
    "write a poem",
    "explain computer engineering",
    "what is artificial intelligence",
    "what is machine learning",
    "what is the meaning of life",
    "help me with my mathematics",
    "solve this algebra question",
    "calculate five plus seven",
    "what is the square root of sixteen",
    "who is the president",
    "what is the weather today",
    "play some music",
    "open a website",
    "send a message to my friend",
    "call my brother",
    "what time is it",
    "show me the news",
    "how does electricity work",
    "explain neural networks",
    "what is a transistor",
    "teach me python programming",
    "write a python program",
    "create a website",
    "make a presentation",
    "I am hungry",
    "I want to sleep",
    "I need some water",
    "where is my phone",
    "I have an examination tomorrow",
    "I am going to school",
    "what should I eat today",
    "tell me about Nigeria",
    "tell me about London",
    "what is two multiplied by eight",
    "how do I become successful",
    "I want to start a business",
    "help me study for my examination",
    "what is the capital of France",
    "explain the difference between hardware and software",
    "I like mathematics",
    "I am feeling tired",
    "what is the distance to the moon",
    "write an email for me",
    "summarize this paragraph",
    "translate this sentence",
    "what is the latest technology",
    "who invented the computer",
    "explain the internet",
    "how do I learn electronics",
    "what is a database",
    "how do I build an application",
    "make a drawing",
    "show me a photograph",
    "what is the meaning of this word",
    "I need help with my assignment",
    "how many days are in a year",
    "what is the population of Nigeria",
    "what is the difference between voltage and current",
    "tell me something interesting",
    "I want to learn artificial intelligence",
    "what is the best programming language",
    "how do I install tensorflow",
    "what is a microcontroller",
    "explain the Arduino Uno",
    "how does a stepper motor work",
    "what is an LED",
    "what is an RGB colour",
    "what is an eight by eight display",
    "I want to buy a laptop",
    "recommend a book",
    "how do I improve my productivity",
    "what is the answer to this question",
    "please wait for me",
    "I will come back later",
    "I am working on a project",
    "can you explain this topic",
    "what does this error mean",
    "help me debug my code",
    "how do I connect to wifi",
    "what is bluetooth",
    "how do I use a keyboard",
    "what is a touchscreen",
    "make the room comfortable",
    "turn on the television",
    "switch off the fan",
    "open the door",
    "close the window",
    "increase the volume",
    "decrease the brightness",
    "turn on the house lights",
    "turn off the kitchen light",
    "change my phone wallpaper",
    "set an alarm for tomorrow",
    "remind me to study",
]

unknown_test = [
    "my name is martel",
    "my favourite subject is mathematics",
    "explain photosynthesis",
    "write a short story about a robot",
    "what is the capital of Ghana",
    "calculate the derivative of x squared",
    "I need to charge my phone",
    "please open the browser",
    "turn on the television",
    "switch off the ceiling fan",
    "make the room brighter",
    "play a song for me",
    "send an email",
    "what is the latest news",
    "I want to learn JavaScript",
    "how do I build a mobile application",
    "explain the theory of relativity",
    "what is the square root of 144",
    "tell me a joke about computers",
    "I am going to class",
    "I need help with my assignment",
    "what is the weather in Lagos",
    "how do I connect a keyboard",
    "show me a picture of a cat",
    "write a business plan",
    "what is the difference between AI and ML",
    "help me prepare for an examination",
    "turn on the bedroom light",
    "increase the television volume",
    "what is the meaning of success",
]


# ============================================================
# 4. TEXT PROCESSING
# ============================================================

TOKEN_PATTERN = r"[a-z0-9]+"

def tokenize(text):
    return re.findall(TOKEN_PATTERN, str(text).lower())


def normalize(text):
    return " ".join(tokenize(text))


# Check duplicate phrases across supported classes.
phrase_owner = {}

for intent, phrases in training_data.items():
    for phrase in phrases:
        key = normalize(phrase)

        if key in phrase_owner and phrase_owner[key] != intent:
            raise ValueError(
                f"Conflicting training phrase: {phrase!r}"
            )

        phrase_owner[key] = intent


# Remove duplicate phrases within the same class.
for intent in REQUIRED_CLASSES:
    unique = {}
    for phrase in training_data[intent]:
        unique[normalize(phrase)] = phrase
    training_data[intent] = list(unique.values())


classes = REQUIRED_CLASSES + [UNKNOWN_CLASS]
class_to_id = {
    name: index for index, name in enumerate(classes)
}


train_texts = []
train_labels = []

for intent in REQUIRED_CLASSES:
    for phrase in training_data[intent]:
        train_texts.append(phrase)
        train_labels.append(class_to_id[intent])

for phrase in unknown_training:
    train_texts.append(phrase)
    train_labels.append(class_to_id[UNKNOWN_CLASS])


# The vocabulary is built only from training phrases.
vocabulary = sorted({
    word
    for phrase in train_texts
    for word in tokenize(phrase)
})

word_to_id = {
    word: index for index, word in enumerate(vocabulary)
}


def encode_text(text):
    vector = np.zeros(len(vocabulary), dtype=np.float32)

    for word in tokenize(text):
        index = word_to_id.get(word)

        if index is not None:
            vector[index] = 1.0

    return vector


def token_coverage(text):
    tokens = tokenize(text)

    if not tokens:
        return 0.0

    known = sum(
        1 for token in tokens
        if token in word_to_id
    )

    return known / len(tokens)


X = np.asarray(
    [encode_text(text) for text in train_texts],
    dtype=np.float32
)

y = np.asarray(train_labels, dtype=np.int32)


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 5. NEURAL NETWORK
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(len(vocabulary),),
        name="word_presence"
    ),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(
        len(classes),
        activation="softmax",
        name="intent_scores"
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=0.00005
    ),
]


print("\n====================================")
print("AIEng4D INTENT MODEL V3")
print("====================================")
print("Supported intents:", len(REQUIRED_CLASSES))
print("Total classes including UNKNOWN:", len(classes))
print("Training examples:", len(X))
print("Vocabulary size:", len(vocabulary))
print("Training split:", len(X_train))
print("Validation split:", len(X_val))

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=180,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 6. PARAMETER EXTRACTION
# ============================================================

# RGB colour aliases.
# The actual RGB values are sent to the firmware as parameters.
# Physical RGB LED wiring and common-anode/common-cathode behaviour
# must be checked when implementing the hardware controller.

RGB_COLOURS = {
    "red": {
        "aliases": ["red"],
        "rgb": [255, 0, 0]
    },
    "green": {
        "aliases": ["green"],
        "rgb": [0, 255, 0]
    },
    "blue": {
        "aliases": ["blue"],
        "rgb": [0, 0, 255]
    },
    "yellow": {
        "aliases": ["yellow"],
        "rgb": [255, 255, 0]
    },
    "cyan": {
        "aliases": ["cyan", "aqua"],
        "rgb": [0, 255, 255]
    },
    "white": {
        "aliases": ["white"],
        "rgb": [255, 255, 255]
    },
    "orange": {
        "aliases": ["orange"],
        "rgb": [255, 128, 0]
    },
    "pink": {
        "aliases": ["pink"],
        "rgb": [255, 105, 180]
    },
    "magenta": {
        "aliases": ["magenta"],
        "rgb": [255, 0, 255]
    },
    "purple": {
        "aliases": ["purple", "violet"],
        "rgb": [128, 0, 255]
    },
}


MATRIX_SYMBOLS = {
    "smiley": ["smiley", "smile", "happy face", "happy"],
    "sad_face": ["sad face", "sad"],
    "heart": ["heart"],
    "star": ["star"],
    "up_arrow": ["up arrow"],
    "down_arrow": ["down arrow"],
    "left_arrow": ["left arrow"],
    "right_arrow": ["right arrow"],
    "tick": ["tick", "check mark", "confirmation"],
    "cross": ["cross", "x symbol"],
    "warning": ["warning", "warning symbol"],
    "question": ["question mark"],
    "letter_a": ["letter a"],
    "letter_b": ["letter b"],
    "letter_c": ["letter c"],
    "letter_z": ["letter z"],
    "number_3": ["number 3"],
    "number_5": ["number 5"],
    "number_7": ["number 7"],
    "number_8": ["number 8"],
}


def contains_any(text, phrases):
    text = normalize(text)

    for phrase in phrases:
        phrase = normalize(phrase)

        if not phrase:
            continue

        if re.search(
            r"(?<![a-z0-9])"
            + re.escape(phrase)
            + r"(?![a-z0-9])",
            text
        ):
            return True

    return False


def find_matches(text, alias_map):
    matches = []

    for canonical, aliases in alias_map.items():
        if contains_any(text, aliases):
            matches.append(canonical)

    return matches


def has_negation(text):
    patterns = [
        r"\bnot\b",
        r"\bnever\b",
        r"\bno\b",
        r"\bdon't\b",
        r"\bdo not\b",
        r"\bdoes not\b",
        r"\bshouldn't\b",
        r"\bshould not\b",
        r"\bcan't\b",
        r"\bcannot\b",
    ]

    lowered = text.lower()

    return any(re.search(p, lowered) for p in patterns)


def extract_motor_angle(text):
    """
    Extract exactly one angle.

    Accepted examples:
        move motor to 90 degrees
        set motor angle to 45
        move stepper to angle 120

    Rejects:
        move motor somewhere
        move motor to 900 degrees
        move motor to 90 and 180 degrees
    """

    lowered = text.lower()

    number_pattern = r"-?\d+(?:\.\d+)?"

    patterns = [
        rf"\b(?:to|at|of)\s+({number_pattern})\s*(?:degrees?|deg)\b",
        rf"\bangle\s*(?:of|to|is)?\s*({number_pattern})\b",
        rf"\bposition\s*(?:of|to|is)?\s*({number_pattern})\s*(?:degrees?|deg)?\b",
        rf"\btarget\s*(?:of|to|is)?\s*({number_pattern})\s*(?:degrees?|deg)?\b",
    ]

    values = []

    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass

    unique_values = list(dict.fromkeys(values))

    if len(unique_values) != 1:
        return None

    angle = unique_values[0]

    if angle < MOTOR_MIN_DEGREES:
        return None

    if angle > MOTOR_MAX_DEGREES:
        return None

    if not angle.is_integer():
        return None

    return int(angle)


def extract_matrix_symbol(text):
    matches = []

    for canonical, aliases in MATRIX_SYMBOLS.items():
        if contains_any(text, aliases):
            matches.append(canonical)

    # Only accept exactly one recognized symbol.
    if len(matches) != 1:
        return None

    return matches[0]


def parse_parameters(intent, text):
    """
    Return a parameter dictionary only when the command is complete
    and unambiguous. Otherwise return None.
    """

    if has_negation(text):
        return None

    lowered = normalize(text)

    if intent == "LED_ON":
        if not contains_any(
            lowered,
            ["led", "indicator", "status light", "board led"]
        ):
            return None

        if contains_any(
            lowered,
            ["off", "switch off", "turn off", "deactivate", "disable"]
        ):
            return None

        if not contains_any(
            lowered,
            [
                "on", "turn on", "switch on", "activate",
                "enable", "illuminate", "glow", "shine",
                "light up", "power on"
            ]
        ):
            return None

        return {}

    if intent == "LED_OFF":
        if not contains_any(
            lowered,
            ["led", "indicator", "status light", "board led"]
        ):
            return None

        if contains_any(
            lowered,
            ["turn on", "switch on", "activate", "enable", "glow"]
        ):
            return None

        if not contains_any(
            lowered,
            [
                "off", "turn off", "switch off", "deactivate",
                "disable", "extinguish", "dark", "power down",
                "stop the led", "stop the indicator"
            ]
        ):
            return None

        return {}

    if intent == "RGB_SET":
        if not contains_any(
            lowered,
            ["rgb", "rgb led", "rgb module", "rgb light"]
        ):
            return None

        matches = find_matches(text, {
            name: data["aliases"]
            for name, data in RGB_COLOURS.items()
        })

        if len(matches) != 1:
            return None

        colour = matches[0]

        return {
            "colour": colour,
            "rgb": RGB_COLOURS[colour]["rgb"]
        }

    if intent == "MOTOR_MOVE":
        if not contains_any(
            lowered,
            ["motor", "stepper", "shaft"]
        ):
            return None

        angle = extract_motor_angle(text)

        if angle is None:
            return None

        return {
            "angle_degrees": angle
        }

    if intent == "MOTOR_CW":
        if not contains_any(
            lowered,
            ["motor", "stepper", "shaft"]
        ):
            return None

        if not contains_any(
            lowered,
            ["clockwise", "clock wise"]
        ):
            return None

        return {
            "direction": "clockwise"
        }

    if intent == "MOTOR_CCW":
        if not contains_any(
            lowered,
            ["motor", "stepper", "shaft"]
        ):
            return None

        if not contains_any(
            lowered,
            [
                "counterclockwise",
                "counter clockwise",
                "anticlockwise",
                "anti clockwise"
            ]
        ):
            return None

        return {
            "direction": "counterclockwise"
        }

    if intent == "MOTOR_STOP":
        if not contains_any(
            lowered,
            ["motor", "stepper", "shaft"]
        ):
            return None

        if not contains_any(
            lowered,
            [
                "stop", "halt", "pause", "cancel",
                "cease", "stationary", "still"
            ]
        ):
            return None

        return {
            "action": "stop"
        }

    if intent == "MATRIX_SHOW":
        if not contains_any(
            lowered,
            ["matrix", "8x8", "8 by 8", "dot display", "led panel"]
        ):
            return None

        symbol = extract_matrix_symbol(text)

        if symbol is None:
            return None

        return {
            "symbol": symbol
        }

    if intent == "MATRIX_CLEAR":
        if not contains_any(
            lowered,
            ["matrix", "8x8", "8 by 8", "dot display", "led panel"]
        ):
            return None

        if not contains_any(
            lowered,
            [
                "clear", "blank", "erase", "remove",
                "reset", "wipe", "empty", "dark"
            ]
        ):
            return None

        return {
            "action": "clear"
        }

    return None


# ============================================================
# 7. COMMAND CLASSIFICATION AND REJECTION
# ============================================================

def classify_command(text):
    """
    Full command pipeline.

    A command is accepted only if:
      1. It is not empty.
      2. It has adequate vocabulary coverage.
      3. The classifier predicts a supported intent.
      4. Confidence passes the initial threshold.
      5. The top-two score margin passes the threshold.
      6. Deterministic parameter extraction succeeds.

    The classifier is never allowed to directly execute hardware.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "accepted": False,
            "reason": "Empty input"
        }

    coverage = token_coverage(text)

    if coverage < MIN_TOKEN_COVERAGE:
        return {
            "accepted": False,
            "reason": "Insufficient known vocabulary",
            "token_coverage": coverage
        }

    vector = encode_text(text).reshape(1, -1)
    scores = model.predict(vector, verbose=0)[0]

    order = np.argsort(scores)[::-1]

    top_index = int(order[0])
    second_index = int(order[1])

    predicted_intent = classes[top_index]

    confidence = float(scores[top_index])
    margin = float(scores[top_index] - scores[second_index])

    if predicted_intent == UNKNOWN_CLASS:
        return {
            "accepted": False,
            "reason": "Classified as UNKNOWN",
            "confidence": confidence,
            "margin": margin
        }

    if confidence < MIN_CONFIDENCE:
        return {
            "accepted": False,
            "reason": "Confidence below threshold",
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "margin": margin
        }

    if margin < MIN_MARGIN:
        return {
            "accepted": False,
            "reason": "Prediction is ambiguous",
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "margin": margin
        }

    parameters = parse_parameters(
        predicted_intent,
        text
    )

    if parameters is None:
        return {
            "accepted": False,
            "reason": "Missing, invalid or ambiguous command parameters",
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "margin": margin
        }

    return {
        "accepted": True,
        "intent": predicted_intent,
        "parameters": parameters,
        "confidence": confidence,
        "margin": margin,
        "token_coverage": coverage
    }


# ============================================================
# 8. HELD-OUT CLASSIFICATION TEST
# ============================================================

test_texts = []
test_labels = []

for intent in REQUIRED_CLASSES:
    for phrase in test_data[intent]:
        test_texts.append(phrase)
        test_labels.append(class_to_id[intent])

for phrase in unknown_test:
    test_texts.append(phrase)
    test_labels.append(class_to_id[UNKNOWN_CLASS])


X_test = np.asarray(
    [encode_text(text) for text in test_texts],
    dtype=np.float32
)

y_test = np.asarray(test_labels, dtype=np.int32)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

test_scores = model.predict(X_test, verbose=0)
test_predictions = np.argmax(test_scores, axis=1)

report = classification_report(
    y_test,
    test_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=list(range(len(classes)))
)


# ============================================================
# 9. COMMAND PIPELINE TESTS
# ============================================================

pipeline_tests = [
    ("Turn on the indicator LED", "LED_ON", True),
    ("Switch off the indicator LED", "LED_OFF", True),

    ("Set the RGB LED to purple", "RGB_SET", True),
    ("Set RGB to cyan", "RGB_SET", True),
    ("Change the RGB module to orange", "RGB_SET", True),

    ("Move the stepper motor to 90 degrees", "MOTOR_MOVE", True),
    ("Set the motor angle to 180 degrees", "MOTOR_MOVE", True),

    ("Rotate the stepper clockwise", "MOTOR_CW", True),
    ("Rotate the motor counterclockwise", "MOTOR_CCW", True),
    ("Stop the stepper motor", "MOTOR_STOP", True),

    ("Display a heart on the LED matrix", "MATRIX_SHOW", True),
    ("Show the letter A on the matrix", "MATRIX_SHOW", True),
    ("Clear the LED matrix", "MATRIX_CLEAR", True),

    ("My name is Martel", None, False),
    ("Tell me a joke", None, False),
    ("Calculate five plus seven", None, False),
    ("Turn on the television", None, False),

    ("Make it purple", None, False),
    ("Set RGB to purple and orange", None, False),
    ("Move the motor somewhere", None, False),
    ("Move the motor to 900 degrees", None, False),
    ("Move the motor to 90 and 180 degrees", None, False),
    ("Turn on the LED and turn it off", None, False),
    ("Do not turn on the LED", None, False),
    ("Show something on the matrix", None, False),
    ("Show an unsupported symbol on the matrix", None, False),
]

pipeline_lines = []
pipeline_passed = 0

print("\n====================================")
print("COMMAND PIPELINE TESTS")
print("====================================")

for phrase, expected_intent, expected_accepted in pipeline_tests:
    result = classify_command(phrase)

    actual_accepted = result["accepted"]

    if expected_accepted:
        passed = (
            actual_accepted
            and result.get("intent") == expected_intent
        )
    else:
        passed = not actual_accepted

    if passed:
        pipeline_passed += 1

    status = "PASS" if passed else "FAIL"

    line = (
        f"{status:4s} | {phrase:48s} | "
        f"{result.get('intent', result.get('reason'))}"
    )

    print(line)
    pipeline_lines.append(line)

pipeline_accuracy = (
    pipeline_passed / len(pipeline_tests)
) * 100


# ============================================================
# 10. TFLITE INT8 EXPORT
# ============================================================

model_path = OUTPUT_DIR / "intent_model_v3.keras"
tflite_path = OUTPUT_DIR / "intent_model_v3_int8.tflite"


model.save(model_path)


def representative_dataset():
    # Use training examples from all classes.
    # The exact input format is the same multi-hot vector used above.
    sample_count = min(len(X_train), 300)

    for index in range(sample_count):
        yield [X_train[index:index + 1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()

    tflite_path.write_bytes(tflite_model)

    quantization_status = "INT8 conversion successful"

except Exception as error:
    quantization_status = (
        "INT8 conversion failed: " + str(error)
    )

    print("\nWARNING:", quantization_status)
    print("The Keras model has still been saved.")


# ============================================================
# 11. SAVE VOCABULARY, CLASSES AND METADATA
# ============================================================

vocabulary_path = OUTPUT_DIR / "vocabulary_v3.json"
classes_path = OUTPUT_DIR / "classes_v3.json"
metadata_path = OUTPUT_DIR / "metadata_v3.json"
parser_path = OUTPUT_DIR / "command_parser_config.json"

vocabulary_path.write_text(
    json.dumps(vocabulary, indent=2),
    encoding="utf-8"
)

classes_path.write_text(
    json.dumps(classes, indent=2),
    encoding="utf-8"
)


parser_config = {
    "rgb_colours": RGB_COLOURS,
    "matrix_symbols": MATRIX_SYMBOLS,
    "motor_min_degrees": MOTOR_MIN_DEGREES,
    "motor_max_degrees": MOTOR_MAX_DEGREES,
    "requires_explicit_device_context": True,
    "requires_valid_parameters": True,
    "negation_rejection": True,
    "multiple_parameter_rejection": True,
}

parser_path.write_text(
    json.dumps(parser_config, indent=2),
    encoding="utf-8"
)


metadata = {
    "model_name": MODEL_NAME,
    "version": 3,
    "input_type": "binary_bag_of_words",
    "tokenizer_pattern": TOKEN_PATTERN,
    "input_size": len(vocabulary),
    "vocabulary_file": vocabulary_path.name,
    "classes": classes,
    "seed": SEED,
    "training_examples": len(X),
    "validation_examples": len(X_val),
    "held_out_test_examples": len(X_test),
    "classifier_test_accuracy": float(test_accuracy),
    "pipeline_test_count": len(pipeline_tests),
    "pipeline_tests_passed": pipeline_passed,
    "pipeline_test_accuracy": float(pipeline_accuracy),
    "initial_rejection_thresholds": {
        "minimum_confidence": MIN_CONFIDENCE,
        "minimum_margin": MIN_MARGIN,
        "minimum_token_coverage": MIN_TOKEN_COVERAGE
    },
    "motor_range_placeholder": {
        "minimum_degrees": MOTOR_MIN_DEGREES,
        "maximum_degrees": MOTOR_MAX_DEGREES,
        "warning": (
            "Verify the actual motor and mechanical range before "
            "using this value in hardware firmware."
        )
    },
    "quantization": quantization_status,
    "deployment_warning": (
        "Reproduce the tokenizer, vocabulary ordering, multi-hot "
        "encoding and quantization parameters exactly in ESP32 firmware. "
        "Test the TFLite model on the target TFLM port."
    )
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)


# ============================================================
# 12. WRITE FULL EVALUATION REPORT
# ============================================================

results_path = RESULTS_DIR / "intent_v3_results.txt"

with results_path.open("w", encoding="utf-8") as file:
    file.write("AIEng4D INTENT MODEL V3 RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Supported intents: {len(REQUIRED_CLASSES)}\n")
    file.write(f"Total classes: {len(classes)}\n")
    file.write(f"Training examples: {len(X)}\n")
    file.write(f"Vocabulary size: {len(vocabulary)}\n")
    file.write(f"Held-out examples: {len(X_test)}\n")
    file.write(
        f"Classifier test accuracy: {test_accuracy * 100:.2f}%\n"
    )

    file.write("\nCLASSIFICATION REPORT\n")
    file.write(report)

    file.write("\nCONFUSION MATRIX\n")
    file.write(np.array2string(matrix))

    file.write("\n\nCOMMAND PIPELINE TESTS\n")
    file.write("\n".join(pipeline_lines))

    file.write("\n\nPIPELINE TEST SUMMARY\n")
    file.write(
        f"Passed: {pipeline_passed}/{len(pipeline_tests)}\n"
    )
    file.write(
        f"Pipeline test score: {pipeline_accuracy:.2f}%\n"
    )

    file.write("\n\nMODEL FILE SIZES\n")
    file.write(
        f"Keras model: {model_path.stat().st_size:,} bytes\n"
    )

    if tflite_path.exists():
        file.write(
            f"INT8 TFLite: {tflite_path.stat().st_size:,} bytes\n"
        )

    file.write("\n\nQUANTIZATION\n")
    file.write(quantization_status)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print(f"Classifier test accuracy: {test_accuracy * 100:.2f}%")
print(
    f"Command pipeline tests: "
    f"{pipeline_passed}/{len(pipeline_tests)} "
    f"({pipeline_accuracy:.2f}%)"
)

print("\nSaved files:")
print("Keras model:", model_path)
print("TFLite model:", tflite_path)
print("Vocabulary:", vocabulary_path)
print("Classes:", classes_path)
print("Metadata:", metadata_path)
print("Parser configuration:", parser_path)
print("Evaluation:", results_path)

print("\nQuantization:", quantization_status)

print("\nIMPORTANT:")
print("A passing software test does not prove safe hardware operation.")
print("Verify the motor range, RGB wiring and matrix driver separately.")
print("Do not execute commands directly from neural-network output.")
