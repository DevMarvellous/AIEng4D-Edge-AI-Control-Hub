
"""
AIEng4D - Intent Classification and Command Validation

Run from the project root:
    python scripts/train_intent.py

This script is self-contained. It does not require an external dataset file.

Outputs:
    models/intent/intent_model.keras
    models/intent/intent_model_int8.tflite
    models/intent/vocabulary.json
    models/intent/classes.json
    models/intent/metadata.json
    models/intent/command_parser_config.json
    results/intent_results.txt

Architecture:
- Neural network classifies the requested hardware action.
- Deterministic parsing extracts and validates parameters.
- Unsupported, incomplete, negated and ambiguous commands are rejected.
- This script does not control hardware.
"""

import json
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

OUTPUT_DIR = Path("models/intent")
RESULTS_DIR = Path("results")

MODEL_NAME = "AIEng4D Intent Classifier"

MIN_CONFIDENCE = 0.75
MIN_MARGIN = 0.15
MIN_TOKEN_COVERAGE = 0.35

# Software placeholder only.
# Verify the actual motor and mechanical range before deployment.
MOTOR_MIN_DEGREES = 0
MOTOR_MAX_DEGREES = 360

UNKNOWN_CLASS = "UNKNOWN"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INTENT DATASET
# ============================================================

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


# Reconstructed training dataset.
# These examples are not claimed to be the original missing dataset.

training_data = {

    "LED_ON": [
        "turn on the led",
        "switch on the led",
        "activate the led",
        "light up the led",
        "please turn the led on",
        "enable the led",
        "make the led glow",
        "power on the led",
        "I want the led on",
        "turn the indicator led on",
        "switch the indicator led on",
        "activate the indicator led",
        "illuminate the led",
        "let the led shine",
        "start the led",
        "make the led light up",
        "turn on my led",
        "please activate the led",
        "can you switch on the led",
        "keep the led illuminated",
    ],

    "LED_OFF": [
        "turn off the led",
        "switch off the led",
        "deactivate the led",
        "put out the led",
        "please turn the led off",
        "disable the led",
        "stop the led from glowing",
        "power off the led",
        "I want the led off",
        "turn the indicator led off",
        "switch the indicator led off",
        "deactivate the indicator led",
        "extinguish the led",
        "stop the led",
        "shut down the led",
        "make the led go dark",
        "turn off my led",
        "please deactivate the led",
        "can you switch off the led",
        "keep the led unlit",
    ],

    "RGB_SET": [
        "set the rgb to red",
        "change the rgb to blue",
        "make the rgb green",
        "set rgb colour to purple",
        "change the rgb colour to cyan",
        "set the rgb led to orange",
        "make the rgb light yellow",
        "display red on the rgb",
        "show blue using the rgb led",
        "change the light colour to green on rgb",
        "set the rgb colour to white",
        "make the rgb glow pink",
        "change rgb to magenta",
        "set the rgb light to turquoise",
        "display orange on the rgb led",
        "I want purple on the rgb",
        "please set the rgb to yellow",
        "change the rgb led colour to white",
        "make the rgb display cyan",
        "set the rgb light to violet",
    ],

    "MATRIX_SHOW": [
        "show a heart on the matrix",
        "display a heart on the matrix",
        "draw a heart on the matrix",
        "show a smiley face on the matrix",
        "display a happy face on the matrix",
        "show a sad face on the matrix",
        "display the letter A on the matrix",
        "show letter B on the matrix",
        "display the number 3 on the matrix",
        "show number 5 on the matrix",
        "draw an arrow on the matrix",
        "show an upward arrow on the matrix",
        "display a downward arrow on the matrix",
        "show a check mark on the matrix",
        "display a cross symbol on the matrix",
        "show a star on the matrix",
        "display a warning symbol on the matrix",
        "show a question mark on the matrix",
        "draw a square on the matrix",
        "display a diamond shape on the matrix",
    ],

    "MATRIX_CLEAR": [
        "clear the matrix",
        "erase the matrix display",
        "blank the matrix",
        "remove the matrix pattern",
        "clear the led matrix",
        "make the matrix blank",
        "turn off the matrix display",
        "reset the matrix",
        "wipe the matrix screen",
        "empty the matrix display",
        "remove everything from the matrix",
        "clear the 8 by 8 display",
        "make the display go dark",
        "erase everything on the matrix",
        "blank out the led matrix",
        "stop displaying the pattern on the matrix",
        "clear the dot display",
        "remove the displayed symbol from the matrix",
        "switch off the matrix display",
        "reset the led display",
    ],

    "MOTOR_MOVE": [
        "move the motor to 90 degrees",
        "rotate the motor to 180 degrees",
        "set the motor position to 45 degrees",
        "move the motor to 120 degrees",
        "rotate the motor to 270 degrees",
        "move the motor to 30 degrees",
        "set the motor angle to 60 degrees",
        "position the motor at 150 degrees",
        "move the motor to 200 degrees",
        "rotate the motor to 300 degrees",
        "set the motor to 10 degrees",
        "move the motor to 360 degrees",
        "turn the motor to 75 degrees",
        "position the motor at 225 degrees",
        "rotate the motor to 135 degrees",
        "move the motor to 15 degrees",
        "set the motor position at 240 degrees",
        "move the stepper motor to 180 degrees",
        "rotate the stepper motor to 90 degrees",
        "set the motor angle to 315 degrees",
    ],

    "MOTOR_CW": [
        "rotate the motor clockwise",
        "turn the motor clockwise",
        "move the motor clockwise",
        "spin clockwise",
        "rotate clockwise",
        "run the motor clockwise",
        "make the motor turn clockwise",
        "move clockwise",
        "start clockwise rotation",
        "rotate the stepper clockwise",
        "turn the stepper motor clockwise",
        "spin the motor in a clockwise direction",
        "please rotate clockwise",
        "activate clockwise rotation",
        "make the motor spin clockwise",
        "rotate the motor CW",
        "turn the motor clockwise now",
        "run clockwise",
        "move the motor in the clockwise direction",
        "begin clockwise movement",
    ],

    "MOTOR_CCW": [
        "rotate the motor counterclockwise",
        "turn the motor counterclockwise",
        "move the motor counterclockwise",
        "spin counterclockwise",
        "rotate counterclockwise",
        "run the motor counterclockwise",
        "make the motor turn counterclockwise",
        "move counterclockwise",
        "start counterclockwise rotation",
        "rotate the stepper counterclockwise",
        "turn the stepper motor anticlockwise",
        "spin the motor in an anticlockwise direction",
        "please rotate anticlockwise",
        "activate counterclockwise rotation",
        "make the motor spin anticlockwise",
        "rotate the motor counterclockwise now",
        "turn the motor anticlockwise now",
        "run counterclockwise",
        "move the motor in the anticlockwise direction",
        "begin counterclockwise movement",
    ],

    "MOTOR_STOP": [
        "stop the motor",
        "halt the motor",
        "stop motor rotation",
        "turn off the motor",
        "pause the motor",
        "cancel motor movement",
        "freeze the motor",
        "stop the stepper motor",
        "halt motor movement",
        "make the motor stop",
        "stop rotating",
        "please stop the motor",
        "end motor movement",
        "disable the motor",
        "stop the rotation",
        "stop the stepper",
        "cancel the rotation",
        "bring the motor to a stop",
        "motor stop",
        "halt the stepper",
    ],
}


# Separate held-out test phrases.
# These are not used for model fitting or vocabulary construction.

test_data = {

    "LED_ON": [
        "please illuminate the led",
        "make my led turn on",
        "I need the led activated",
        "switch the led on now",
        "keep my indicator led on",
    ],

    "LED_OFF": [
        "please extinguish the led",
        "make my led turn off",
        "I need the led deactivated",
        "switch the led off now",
        "disable my indicator led",
    ],

    "RGB_SET": [
        "set the rgb led colour to red",
        "make the rgb light purple",
        "display cyan with the rgb",
        "change the rgb to pink",
        "I want the rgb to glow blue",
    ],

    "MATRIX_SHOW": [
        "put a heart on the display matrix",
        "draw a smiley on the matrix",
        "display letter C on the matrix",
        "show an arrow on the led matrix",
        "put a star on the display matrix",
    ],

    "MATRIX_CLEAR": [
        "remove the pattern from the matrix display",
        "wipe the led matrix clean",
        "make the 8 by 8 screen empty",
        "erase the symbol from the matrix display",
        "turn the led matrix display dark",
    ],

    "MOTOR_MOVE": [
        "position the motor at 80 degrees",
        "move the stepper to 210 degrees",
        "rotate the motor to 100 degrees",
        "set the motor angle to 330 degrees",
        "position the motor at 25 degrees",
    ],

    "MOTOR_CW": [
        "start rotating in the clockwise direction",
        "make the stepper spin clockwise",
        "begin turning clockwise",
        "rotate the motor clockwise now",
    ],

    "MOTOR_CCW": [
        "start rotating in the counterclockwise direction",
        "make the stepper spin anticlockwise",
        "begin turning counterclockwise",
        "rotate the motor anticlockwise now",
    ],

    "MOTOR_STOP": [
        "stop all motor movement",
        "please halt rotation",
        "cancel the current motor operation",
        "bring the stepper to a halt",
        "stop the motor immediately",
    ],
}


# ============================================================
# 3. UNKNOWN TRAINING DATA
# ============================================================

unknown_training = [
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
# 4. TEXT PROCESSING AND DATA VALIDATION
# ============================================================

TOKEN_PATTERN = r"[a-z0-9]+"


def tokenize(text):
    return re.findall(TOKEN_PATTERN, str(text).lower())


def normalize(text):
    return " ".join(tokenize(text))


if list(training_data.keys()) != REQUIRED_CLASSES:
    raise ValueError("Training dataset class order is invalid.")

if list(test_data.keys()) != REQUIRED_CLASSES:
    raise ValueError("Test dataset class order is invalid.")


# Remove duplicate training phrases within each supported class.
for intent in REQUIRED_CLASSES:
    unique = {}

    for phrase in training_data[intent]:
        key = normalize(phrase)

        if not key:
            raise ValueError(f"Empty training phrase in {intent}.")

        unique[key] = phrase

    training_data[intent] = list(unique.values())


# Check that no training phrase belongs to multiple supported classes.
phrase_owner = {}

for intent, phrases in training_data.items():
    for phrase in phrases:
        key = normalize(phrase)

        if key in phrase_owner and phrase_owner[key] != intent:
            raise ValueError(
                f"Conflicting training phrase: {phrase!r}"
            )

        phrase_owner[key] = intent


# Check exact phrase overlap between training and held-out test data.
supported_train_keys = {
    normalize(phrase)
    for phrases in training_data.values()
    for phrase in phrases
}

supported_test_keys = {
    normalize(phrase)
    for phrases in test_data.values()
    for phrase in phrases
}

supported_overlap = supported_train_keys & supported_test_keys

if supported_overlap:
    raise ValueError(
        "Supported training/test overlap: "
        + ", ".join(sorted(supported_overlap))
    )


# Deduplicate UNKNOWN data and remove the old duplicate.
unknown_training = list({
    normalize(phrase): phrase
    for phrase in unknown_training
}.values())

unknown_test = list({
    normalize(phrase): phrase
    for phrase in unknown_test
}.values())

unknown_train_keys = {
    normalize(phrase) for phrase in unknown_training
}

unknown_test_keys = {
    normalize(phrase) for phrase in unknown_test
}

unknown_overlap = unknown_train_keys & unknown_test_keys

if unknown_overlap:
    raise ValueError(
        "UNKNOWN training/test overlap: "
        + ", ".join(sorted(unknown_overlap))
    )


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


# Split raw phrases BEFORE vocabulary construction.
# This prevents validation-only words from entering the vocabulary.

y = np.asarray(train_labels, dtype=np.int32)

all_indices = np.arange(len(train_texts))

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# Build vocabulary using training phrases ONLY.
vocabulary = sorted({
    word
    for index in train_indices
    for word in tokenize(train_texts[index])
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

X_train = X[train_indices]
X_val = X[val_indices]

y_train = y[train_indices]
y_val = y[val_indices]


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
print("AIEng4D INTENT CLASSIFIER")
print("====================================")
print("Supported intents:", len(REQUIRED_CLASSES))
print("Total classes including UNKNOWN:", len(classes))
print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Vocabulary size:", len(vocabulary))

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
        r"\bdoesn't\b",
        r"\bdoes not\b",
        r"\bshouldn't\b",
        r"\bshould not\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\bwon't\b",
        r"\bwill not\b",
    ]

    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in patterns
    )


def extract_motor_angle(text):
    """
    Extract exactly one integer angle.

    Rejects missing, multiple, decimal and out-of-range angles.
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
            values.append(float(match.group(1)))

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

        matches = find_matches(
            text,
            {
                name: data["aliases"]
                for name, data in RGB_COLOURS.items()
            }
        )

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

        if contains_any(
            lowered,
            [
                "clockwise", "clock wise",
                "counterclockwise", "counter clockwise",
                "anticlockwise", "anti clockwise",
                "stop", "halt", "pause", "cancel"
            ]
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

        if contains_any(
            lowered,
            [
                "counterclockwise", "counter clockwise",
                "anticlockwise", "anti clockwise",
                "stop", "halt", "pause", "cancel"
            ]
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
                "counterclockwise", "counter clockwise",
                "anticlockwise", "anti clockwise"
            ]
        ):
            return None

        if contains_any(
            lowered,
            ["clockwise", "clock wise", "stop", "halt", "pause", "cancel"]
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

        if contains_any(
            lowered,
            [
                "clear", "blank", "erase", "remove",
                "reset", "wipe", "empty", "dark"
            ]
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

        if extract_matrix_symbol(text) is not None:
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
    A command is accepted only if:
      1. Input is not empty.
      2. It is not negated.
      3. Vocabulary coverage is adequate.
      4. The predicted class is supported.
      5. Confidence passes the threshold.
      6. Top-two score margin passes the threshold.
      7. Parameter extraction succeeds.

    The classifier never directly executes hardware.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "accepted": False,
            "reason": "Empty input"
        }

    if has_negation(text):
        return {
            "accepted": False,
            "reason": "Negated command rejected"
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

    ("Clear the matrix and show a heart", None, False),
    ("Rotate the motor clockwise and counterclockwise", None, False),
    ("Move the motor to 90 degrees clockwise", None, False),
    ("Stop the motor and rotate clockwise", None, False),
    ("Turn on the LED and turn it off", None, False),
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
# 10. TFLITE INT8 EXPORT AND EVALUATION
# ============================================================

model_path = OUTPUT_DIR / "intent_model.keras"
tflite_path = OUTPUT_DIR / "intent_model_int8.tflite"

model.save(model_path)

quantization_status = "INT8 conversion not completed"

tflite_accuracy = None
tflite_agreement = None

tflite_input_scale = None
tflite_input_zero_point = None
tflite_output_scale = None
tflite_output_zero_point = None

tflite_error = None

# Remove an older model so a failed conversion cannot leave
# a stale file that appears to be the newly generated model.

if tflite_path.exists():
    tflite_path.unlink()


def representative_dataset():
    rng = np.random.default_rng(SEED)

    sample_count = min(len(X_train), 300)

    selected = rng.choice(
        len(X_train),
        size=sample_count,
        replace=False
    )

    for index in selected:
        yield [
            X_train[index:index + 1].astype(np.float32)
        ]


try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_model
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("Invalid INT8 quantization parameters.")

    tflite_input_scale = float(input_scale)
    tflite_input_zero_point = int(input_zero_point)

    tflite_output_scale = float(output_scale)
    tflite_output_zero_point = int(output_zero_point)

    quantized_predictions = []

    for sample in X_test:
        input_float = sample.reshape(1, -1)

        input_quantized = np.round(
            input_float / input_scale + input_zero_point
        )

        input_quantized = np.clip(
            input_quantized,
            -128,
            127
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            input_quantized
        )

        interpreter.invoke()

        output_quantized = interpreter.get_tensor(
            output_details["index"]
        )

        output_float = (
            output_quantized.astype(np.float32)
            - output_zero_point
        ) * output_scale

        predicted_class = int(
            np.argmax(output_float[0])
        )

        quantized_predictions.append(predicted_class)

    quantized_predictions = np.asarray(
        quantized_predictions,
        dtype=np.int32
    )

    tflite_accuracy = float(
        np.mean(quantized_predictions == y_test)
    )

    tflite_agreement = float(
        np.mean(quantized_predictions == test_predictions)
    )

    quantization_status = (
        "INT8 conversion and held-out evaluation successful"
    )

except Exception as error:
    tflite_error = str(error)

    quantization_status = (
        "INT8 conversion/evaluation failed: " + tflite_error
    )

    if tflite_path.exists():
        tflite_path.unlink()

    print("\nWARNING:", quantization_status)


# ============================================================
# 11. SAVE VOCABULARY, CLASSES AND METADATA
# ============================================================

vocabulary_path = OUTPUT_DIR / "vocabulary.json"
classes_path = OUTPUT_DIR / "classes.json"
metadata_path = OUTPUT_DIR / "metadata.json"
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
    "minimum_confidence": MIN_CONFIDENCE,
    "minimum_margin": MIN_MARGIN,
    "minimum_token_coverage": MIN_TOKEN_COVERAGE,
}

parser_path.write_text(
    json.dumps(parser_config, indent=2),
    encoding="utf-8"
)


metadata = {
    "model_name": MODEL_NAME,
    "version": 2,

    "dataset_description": (
        "Self-contained reconstructed intent dataset. "
        "The original external dataset file was unavailable."
    ),

    "input_type": "binary_bag_of_words",
    "tokenizer_pattern": TOKEN_PATTERN,
    "input_size": len(vocabulary),

    "vocabulary_file": vocabulary_path.name,
    "classes_file": classes_path.name,
    "classes": classes,

    "seed": SEED,

    "training_examples": len(X_train),
    "validation_examples": len(X_val),
    "held_out_test_examples": len(X_test),

    "classifier_test_accuracy": float(test_accuracy),

    "pipeline_test_count": len(pipeline_tests),
    "pipeline_tests_passed": pipeline_passed,
    "pipeline_test_accuracy": float(pipeline_accuracy),

    "rejection_thresholds": {
        "minimum_confidence": MIN_CONFIDENCE,
        "minimum_margin": MIN_MARGIN,
        "minimum_token_coverage": MIN_TOKEN_COVERAGE
    },

    "motor_range_placeholder": {
        "minimum_degrees": MOTOR_MIN_DEGREES,
        "maximum_degrees": MOTOR_MAX_DEGREES,
        "warning": (
            "Verify the actual motor, driver, gearing and "
            "mechanical range before hardware deployment."
        )
    },

    "quantization": {
        "status": quantization_status,
        "tflite_test_accuracy": tflite_accuracy,
        "keras_tflite_prediction_agreement": tflite_agreement,
        "input_scale": tflite_input_scale,
        "input_zero_point": tflite_input_zero_point,
        "output_scale": tflite_output_scale,
        "output_zero_point": tflite_output_zero_point,
        "error": tflite_error
    },

    "deployment_warning": (
        "Reproduce the tokenizer, vocabulary ordering, multi-hot "
        "encoding, quantization and rejection rules exactly in ESP32 "
        "firmware. Verify TFLM operator support and measure tensor "
        "arena memory and inference latency on the ESP32-S3."
    )
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)


# ============================================================
# 12. WRITE FULL EVALUATION REPORT
# ============================================================

results_path = RESULTS_DIR / "intent_results.txt"

with results_path.open("w", encoding="utf-8") as file:
    file.write("AIEng4D INTENT CLASSIFIER RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write("Dataset: reconstructed, self-contained dataset\n")
    file.write(f"Supported intents: {len(REQUIRED_CLASSES)}\n")
    file.write(f"Total classes: {len(classes)}\n")
    file.write(f"Training examples: {len(X_train)}\n")
    file.write(f"Validation examples: {len(X_val)}\n")
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

    file.write("\n\nINT8 EVALUATION\n")
    file.write(quantization_status + "\n")

    if tflite_accuracy is not None:
        file.write(
            f"INT8 held-out accuracy: "
            f"{tflite_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Keras/TFLite prediction agreement: "
            f"{tflite_agreement * 100:.2f}%\n"
        )

        file.write(f"Input scale: {tflite_input_scale}\n")
        file.write(f"Input zero point: {tflite_input_zero_point}\n")
        file.write(f"Output scale: {tflite_output_scale}\n")
        file.write(f"Output zero point: {tflite_output_zero_point}\n")


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

if tflite_accuracy is not None:
    print(
        f"INT8 held-out accuracy: "
        f"{tflite_accuracy * 100:.2f}%"
    )

    print(
        f"Keras/TFLite agreement: "
        f"{tflite_agreement * 100:.2f}%"
    )
else:
    print("INT8 evaluation:", quantization_status)

print("\nSaved files:")
print("Keras model:", model_path)
print("TFLite model:", tflite_path)
print("Vocabulary:", vocabulary_path)
print("Classes:", classes_path)
print("Metadata:", metadata_path)
print("Parser configuration:", parser_path)
print("Evaluation:", results_path)

print("\nIMPORTANT:")
print("A passing software test does not prove safe hardware operation.")
print("Verify the motor range, RGB wiring and matrix driver separately.")
print("Do not execute commands directly from neural-network output.")
