"""
AIEng4D - Intent Classification and Command Validation

Run from anywhere (paths are anchored to the project root):
    python scripts/train_intent.py

Self-contained: the dataset lives in this file and no other source script
is read.

Outputs:
    models/intent/intent_model.keras
    models/intent/intent_model_int8.tflite
    models/intent/vocabulary.json
    models/intent/classes.json
    models/intent/metadata.json
    models/intent/command_parser_config.json
    results/intent_results.txt

Design:
- The neural network classifies the requested hardware action
  (eleven supported intents plus an UNKNOWN class).
- Deterministic parsing extracts parameters and enforces the device
  context (right device named, one unambiguous value).
- Domain checks and parameter validation reject unsupported commands.
- Nothing here controls hardware. Never execute neural-network output
  directly; only accepted, validated commands may reach firmware.

The script exits with a non-zero status if any curated safety test in
section 11 fails, so it can be used as a gate before flashing firmware.
"""

import json
import re
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "models" / "intent"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "AIEng4D intent classifier"

# Network size. Chosen on the validation split only (never the test set).
HIDDEN_UNITS = (24, 12)
DROPOUT_RATE = 0

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
tf.keras.utils.set_random_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SUPPORTED-INTENT DATASET
# ============================================================

# Keep phrases specific to the hardware action. In particular:
# - LED_ON/OFF refer to the individual indicator LED only.
# - RGB_SET refers to changing the RGB module's colour.
# - MATRIX_SHOW / MATRIX_CLEAR refer to the 8x8 LED matrix.
# - MOTOR_MOVE requires an explicit target angle.
# - MOTOR_CW/CCW are continuous direction commands.
# - MOTOR_STOP is a separate immediate stop command.
# - BUZZER_PLAY plays one of a fixed set of firmware sound patterns.
# - BUZZER_STOP silences every buzzer.

training_data = {
    "LED_ON": [
        "turn on the indicator led", "switch on the indicator led",
        "turn the red led on", "turn the green led on",
        "turn the blue led on", "switch the led on",
        "activate the indicator led", "power on the indicator",
        "enable the indicator light", "make the indicator led glow",
        "illuminate the indicator led", "set the indicator led to on",
        "please turn on the led", "please switch on the led",
        "I want the indicator led on", "make the led light up",
        "start the indicator led", "the indicator should be on",
        "turn my led indicator on", "light up the status led",
        "enable the status indicator", "switch the status light on",
        "set led state to on", "turn on the small led",
        "activate the green indicator", "activate the red indicator",
        "power up the led indicator", "make the status led shine",
        "I need the led switched on", "led on please", "indicator on",
        "switch the indicator to on", "turn the indicator on now",
        "start the led output", "make the indicator visible",
        "turn on the board led", "switch on the board indicator",
        "turn the status led on", "enable the led", "make the led turn on",
        "light the indicator led", "activate the led",
        "set the status led to on", "get the led glowing",
        "put the indicator light on", "the led should be on",
        "kindly turn the led on", "turn on the led indicator light",
        "I would like the status led on", "illuminate the board led",
        "power the indicator led on", "turn on the status light",
        "make the board led come on", "bring the indicator light on",
        "can you switch the led on", "could you turn on the indicator led",
        "would you please light up the status led",
        "could you turn the indicator on", "make sure the indicator is on",
        "make sure the led is switched on",
        "keep the status led illuminated",
        "the indicator led should be illuminated",
        "start the indicator light",
    ],
    "LED_OFF": [
        "turn off the indicator led", "switch off the indicator led",
        "turn the red led off", "turn the green led off",
        "turn the blue led off", "switch the led off",
        "deactivate the indicator led", "power off the indicator",
        "disable the indicator light", "make the indicator led go dark",
        "set the indicator led to off", "please turn off the led",
        "please switch off the led", "I want the indicator led off",
        "stop the indicator led from glowing", "the indicator should be off",
        "turn my led indicator off", "switch the status light off",
        "disable the status indicator", "set led state to off",
        "turn off the small led", "deactivate the green indicator",
        "deactivate the red indicator", "shut down the led indicator",
        "I need the led switched off", "led off please", "indicator off",
        "switch the indicator to off", "turn the indicator off now",
        "stop the led output", "make the indicator dark",
        "turn off the board led", "extinguish the indicator led",
        "power down the status led", "stop the indicator light",
        "disable the led output", "switch off the board indicator",
        "turn the status led off", "disable the led",
        "make the led turn off", "put out the indicator led",
        "deactivate the led", "set the status led to off",
        "get the led switched off", "put the indicator light off",
        "the led should be off", "kindly turn the led off",
        "turn off the led indicator light",
        "I would like the status led off", "shut off the indicator",
        "make the led stop glowing", "turn off the status light",
        "make the board led go off", "stop the status led from shining",
        "can you switch the led off", "could you turn off the indicator led",
        "would you please switch off the status led",
        "could you turn the indicator off", "make sure the indicator is off",
        "make sure the led is switched off", "extinguish the status led",
        "the board led should be extinguished",
        "keep the status led switched off",
    ],
    "RGB_SET": [
        "set the rgb led to red", "set the rgb led to green",
        "set the rgb led to blue", "change the rgb colour to purple",
        "change the rgb colour to yellow", "change the rgb colour to cyan",
        "set the rgb light to white", "set the rgb light to orange",
        "make the rgb module glow pink", "make the rgb module glow magenta",
        "change the rgb led colour to violet", "set rgb to red",
        "set rgb to green", "set rgb to blue", "set rgb to purple",
        "set rgb to yellow", "set rgb to cyan", "set rgb to white",
        "set rgb to orange", "set rgb to pink", "set rgb to magenta",
        "set rgb to violet", "change the rgb light to red",
        "change the rgb light to green", "change the rgb light to blue",
        "display red on the rgb led", "display blue on the rgb led",
        "show green on the rgb module", "select purple for the rgb led",
        "choose yellow for the rgb light", "make the rgb led white",
        "make the rgb led cyan", "I want a red rgb light",
        "I want a purple rgb light", "change the colour of the rgb module",
        "set the colour on the rgb led", "turn the rgb led red",
        "make the rgb light blue", "switch the rgb module to green",
        "set the rgb colour to orange", "change the rgb to magenta",
        "rgb led purple please", "colour the rgb module yellow",
        "I want the rgb module cyan", "light the rgb led white",
        "set the rgb output to pink", "set the rgb led to aqua",
        "make the rgb module violet", "change the rgb output to red",
        "switch the rgb light to white",
    ],
    "MATRIX_SHOW": [
        "show a smiley on the led matrix",
        "display a heart on the led matrix",
        "show an up arrow on the led matrix",
        "show a down arrow on the led matrix",
        "show a left arrow on the led matrix",
        "show a right arrow on the led matrix",
        "display a star on the led matrix",
        "display a tick on the led matrix", "show a cross on the led matrix",
        "display a warning symbol on the matrix",
        "draw a happy face on the matrix", "draw a sad face on the matrix",
        "put a heart on the matrix", "put a star on the matrix",
        "show the number five on the matrix",
        "show the number eight on the matrix",
        "display a check mark on the matrix",
        "display an arrow on the matrix",
        "show a symbol on the 8 by 8 matrix",
        "draw a pattern on the 8 by 8 matrix",
        "show a smiley on the 8x8 display",
        "display a heart pattern on the 8x8 display",
        "put an arrow on the led panel",
        "show a happy symbol on the led panel",
        "draw a star pattern on the led panel",
        "display a confirmation symbol on the matrix",
        "show a warning sign on the led matrix",
        "render a heart on the matrix",
        "display a simple animation on the matrix",
        "show a face on the led matrix",
        "put a tick symbol on the 8x8 matrix",
        "draw an up arrow on the led panel",
        "display the letter A on the matrix",
        "show the letter B on the matrix",
        "write the number 3 on the matrix",
        "show a question mark on the matrix",
        "show a sad face on the led matrix",
        "display the number 7 on the matrix",
        "show a tick on the 8x8 matrix", "draw a cross on the led matrix",
        "display a right arrow on the matrix",
        "display the number 5 on the led matrix",
        "show a left arrow on the 8x8 panel",
        "draw a smiley face on the dot matrix",
        "put a warning sign on the 8x8 display",
        "display a question mark on the led panel",
        "show a heart on the dot display",
        "draw a star on the 8 by 8 matrix",
        "put the letter A on the led display",
        "show the letter B on the dot panel",
        "display a down arrow on the led display",
        "show an up arrow on the dot panel",
    ],
    "MATRIX_CLEAR": [
        "clear the led matrix", "clear the 8x8 matrix",
        "clear the 8 by 8 display", "blank the led matrix",
        "make the led matrix blank", "erase the matrix pattern",
        "remove the symbol from the matrix",
        "remove the image from the matrix", "clear all pixels on the matrix",
        "turn off all matrix pixels", "make the matrix display empty",
        "reset the led matrix display", "wipe the 8x8 matrix",
        "blank the 8x8 led panel", "erase everything on the led matrix",
        "remove the displayed pattern from the matrix",
        "hide the symbol on the led matrix",
        "clear the current matrix image",
        "stop displaying the matrix pattern",
        "switch off the led matrix pixels", "make the led panel go blank",
        "clear the dot matrix", "reset the dot matrix display",
        "remove the heart from the matrix",
        "remove the arrow from the matrix", "erase the smiley on the matrix",
        "clear the led panel display", "blank the dot display",
        "turn all matrix lights off", "clear the matrix screen",
        "make the 8x8 display dark", "remove everything from the matrix",
        "delete the current matrix pattern", "clear the symbol display",
        "blank the led dot display", "stop the matrix display",
        "clear the matrix", "blank the matrix", "erase the led matrix",
        "reset the matrix", "clear everything from the led panel",
        "make the matrix go dark", "turn off the led matrix",
        "switch off the matrix", "empty the matrix",
        "wipe the dot matrix clean", "clear the display on the matrix",
        "remove the letter from the matrix",
        "take everything off the matrix", "clear the led display",
        "blank the dot panel", "get rid of the pattern on the matrix",
        "erase the content on the matrix",
        "erase the contents of the led panel",
        "remove the content from the 8x8 display",
        "clear the contents of the matrix",
    ],
    "MOTOR_MOVE": [
        "move the stepper motor to 0 degrees",
        "move the stepper motor to 30 degrees",
        "move the stepper motor to 45 degrees",
        "move the stepper motor to 60 degrees",
        "move the stepper motor to 90 degrees",
        "move the stepper motor to 120 degrees",
        "move the stepper motor to 135 degrees",
        "move the stepper motor to 180 degrees",
        "move the stepper motor to 225 degrees",
        "move the stepper motor to 270 degrees",
        "move the stepper motor to 315 degrees",
        "set the motor angle to 45 degrees",
        "set the motor angle to 90 degrees",
        "set the motor angle to 135 degrees",
        "set the motor angle to 180 degrees",
        "set the motor position to 60 degrees",
        "set the motor position to 120 degrees",
        "set the motor position to 270 degrees",
        "position the stepper at 30 degrees",
        "position the stepper at 75 degrees",
        "position the stepper at 150 degrees",
        "position the stepper at 240 degrees",
        "rotate the motor to 45 degrees", "rotate the motor to 90 degrees",
        "rotate the motor to 180 degrees", "rotate the motor to 225 degrees",
        "go to 90 degrees on the motor", "go to 180 degrees on the motor",
        "move the shaft to 120 degrees", "move the shaft to 250 degrees",
        "set the stepper position to 0 degrees",
        "set the stepper position to 300 degrees",
        "I want the motor at 135 degrees", "put the motor at 210 degrees",
        "move the motor to angle 160", "set the motor target to 200 degrees",
        "turn the motor to 150 degrees", "send the stepper to 270 degrees",
        "set the shaft angle to 60 degrees",
        "move the motor shaft to 30 degrees",
        "adjust the stepper to 75 degrees", "point the shaft at 180 degrees",
        "move to angle 45 on the stepper",
        "set the motor angle to 120 degrees",
        "take the stepper to 0 degrees", "bring the motor to 315 degrees",
        "move the stepper motor to 100 degrees",
        "set the stepper to 90 degrees", "place the shaft at 240 degrees",
        "set the motor target to 45 degrees", "turn the motor to angle 120",
        "set the stepper to angle 60", "rotate the shaft to angle 150",
        "turn the stepper to angle 45",
    ],
    "MOTOR_CW": [
        "rotate the stepper motor clockwise", "turn the stepper clockwise",
        "spin the stepper clockwise", "run the stepper clockwise",
        "start clockwise rotation of the motor",
        "move the motor clockwise continuously",
        "rotate clockwise continuously", "make the motor spin clockwise",
        "run the motor in the clockwise direction",
        "start the motor rotating clockwise",
        "turn the motor in a clockwise direction",
        "rotate the shaft clockwise", "spin the shaft clockwise",
        "move the shaft clockwise", "clockwise rotation for the stepper",
        "set motor direction to clockwise",
        "set the stepper direction to clockwise",
        "activate clockwise motor rotation", "continue rotating clockwise",
        "keep rotating clockwise", "make the stepper turn clockwise",
        "begin clockwise movement", "rotate the motor clockwise now",
        "I want clockwise rotation", "start clockwise mode on the motor",
        "run clockwise mode", "turn clockwise continuously",
        "move in the clockwise direction", "rotate in a clockwise direction",
        "spin in the clockwise direction", "clockwise please",
        "motor clockwise", "stepper clockwise", "start clockwise rotation",
        "keep the motor turning clockwise", "drive the motor clockwise",
        "spin the motor clockwise", "make the shaft turn clockwise",
        "rotate the stepper clockwise please", "turn the motor clockwise",
        "keep the stepper spinning clockwise",
        "set the rotation direction to clockwise",
        "make the motor go clockwise",
        "spin the stepper in a clockwise direction",
        "start turning the shaft clockwise",
        "the motor should turn clockwise",
    ],
    "MOTOR_CCW": [
        "rotate the stepper motor counterclockwise",
        "turn the stepper counterclockwise",
        "spin the stepper counterclockwise",
        "run the stepper counterclockwise",
        "start counterclockwise rotation of the motor",
        "move the motor counterclockwise continuously",
        "rotate counterclockwise continuously",
        "make the motor spin counterclockwise",
        "run the motor in the counterclockwise direction",
        "start the motor rotating counterclockwise",
        "turn the motor in a counterclockwise direction",
        "rotate the shaft counterclockwise",
        "spin the shaft counterclockwise", "move the shaft counterclockwise",
        "counterclockwise rotation for the stepper",
        "set motor direction to counterclockwise",
        "set the stepper direction to counterclockwise",
        "activate counterclockwise motor rotation",
        "continue rotating counterclockwise",
        "keep rotating counterclockwise",
        "make the stepper turn counterclockwise",
        "begin counterclockwise movement",
        "rotate the motor counterclockwise now",
        "I want counterclockwise rotation",
        "start counterclockwise mode on the motor",
        "run counterclockwise mode", "turn counterclockwise continuously",
        "move in the counterclockwise direction",
        "rotate in a counterclockwise direction",
        "spin in the counterclockwise direction", "counterclockwise please",
        "motor counterclockwise", "stepper counterclockwise",
        "start counterclockwise rotation",
        "keep the motor turning counterclockwise",
        "drive the motor counterclockwise",
        "spin the motor counterclockwise",
        "make the shaft turn counterclockwise",
        "rotate the stepper counterclockwise please",
        "turn the motor counterclockwise",
        "keep the stepper spinning counterclockwise",
        "set the rotation direction to counterclockwise",
        "rotate the stepper anticlockwise", "spin the motor anti clockwise",
        "turn the shaft anticlockwise", "run the motor anticlockwise",
        "rotate the motor counter clockwise",
        "turn the stepper counter clockwise",
        "make the motor go anti clockwise",
        "spin the shaft counter clockwise",
        "the motor should turn anticlockwise",
        "set the stepper direction to anticlockwise",
    ],
    "MOTOR_STOP": [
        "stop the stepper motor", "stop the motor immediately",
        "halt the stepper motor", "stop motor rotation",
        "stop rotating the motor", "stop the motor movement",
        "bring the stepper to a stop", "halt motor rotation now",
        "pause the stepper motor", "cancel motor movement",
        "stop the shaft from rotating", "stop clockwise rotation",
        "stop counterclockwise rotation", "stop the motor turning",
        "keep the motor stopped", "do not move the stepper",
        "hold the motor still", "end the motor rotation",
        "stop the stepper now", "motor stop", "stepper stop",
        "halt the motor now", "cease motor movement", "cancel the rotation",
        "stop spinning the shaft", "bring the motor movement to an end",
        "stop all stepper movement", "pause motor rotation",
        "stop the motor output", "disable motor movement",
        "I want the motor to stop", "make the stepper stop moving",
        "stop the rotation immediately", "freeze the motor movement",
        "stop the motor and hold position", "emergency stop motor",
        "stop the motor", "stop the stepper", "stop the shaft",
        "halt the shaft", "motor halt", "make the motor stop",
        "make the stepper stationary", "keep the shaft stationary",
        "stop the stepper from turning", "stop turning the motor",
        "stop the shaft rotation", "stop all motor movement",
        "bring the motor to rest", "freeze the stepper", "freeze the shaft",
        "immediately stop the motor", "stop the motor right now",
        "please stop the stepper", "please halt the motor",
        "abort the motor movement", "terminate motor rotation",
        "turn off the stepper motor", "switch off the motor",
        "stop the stepper at its current position",
        "make the shaft stay still", "the motor must stop now",
        "keep the stepper at a standstill", "do not rotate the motor",
        "stop the motor from spinning",
        "stop the stepper rotation right away",
        "cancel the stepper movement", "cancel all motor activity",
        "cancel what the motor is doing", "stop all rotation immediately",
        "halt the rotation", "stop spinning", "stop the stepper activity",
    ],
    "BUZZER_PLAY": [
        "beep", "make a short beep", "play a short beep", "make one beep",
        "play a single beep", "give me a short tone", "make a double beep",
        "play two beeps", "sound two short beeps", "play a double tone",
        "play the success sound", "make a success tone",
        "play a confirmation sound", "give me the success signal",
        "play the error sound", "make an error tone", "play a failure sound",
        "give me a low error beep", "sound the alarm",
        "play an alarm pattern", "start the warning alarm",
        "make the buzzer sound an alarm", "activate the active buzzer",
        "turn on the active buzzer", "start the continuous buzzer alert",
        "sound a steady alert", "play a short tone on the buzzer",
        "make two quick beeps", "play an ascending tone",
        "start the buzzer alarm", "make a quick beep", "play a quick beep",
        "give me a beep", "play a double beep", "make two short beeps",
        "play the success tone", "give me a confirmation tone",
        "play a success signal", "make an ascending tone", "start the alarm",
        "sound the alarm pattern", "play the warning sound",
        "make a warning alarm", "start the repeating warning sound",
        "turn on the continuous buzzer", "switch on the active buzzer",
        "start the steady alert", "activate the continuous alert",
        "sound the continuous buzzer alert", "make the buzzer beep",
        "make a failure tone", "play the error beep",
        "sound the alarm please", "play the alarm now",
        "switch on the steady active buzzer",
        "switch the continuous alert on", "turn the continuous buzzer on",
        "start the alarm right now", "play a short beep now",
    ],
    "BUZZER_STOP": [
        "stop the buzzer", "stop the sound", "silence the buzzer",
        "stop all buzzer sounds", "turn off the active buzzer",
        "deactivate the active buzzer", "switch off the buzzer",
        "stop the alarm", "silence the alarm", "stop the beeping",
        "stop the continuous alert", "turn the buzzer off",
        "disable the buzzer", "halt the buzzer sound",
        "cancel the buzzer alert", "make the buzzer quiet",
        "stop the active alert", "switch the buzzer off now",
        "end the alarm sound", "stop all sounds from the buzzer",
        "silence the active buzzer", "stop the buzzer sound",
        "stop the buzzer now", "mute the buzzer", "turn off the alarm",
        "switch off the alarm", "cancel the alarm", "quiet the alarm",
        "stop the beep", "stop the continuous buzzer",
        "switch off the continuous alert", "please silence the alert",
        "shut off the buzzer", "stop the repeating warning sound",
        "disable the alarm", "stop the steady alert", "make the alarm stop",
        "stop the buzzer sound right now", "stop the alarm now",
        "stop beeping now", "stop the continuous alert now",
        "silence the beeping", "switch the alarm off now",
    ],
}

# Held-out phrases: never used during training or validation.
# Keep this set independent so evaluation is meaningful.
test_data = {
    "LED_ON": [
        "please activate the status led", "make the red indicator turn on",
        "I need the green led illuminated", "enable the board indicator",
        "switch the blue status led into the on state",
        "the indicator light needs to glow", "power up my led indicator",
        "set the status indicator as on", "turn my board led on please",
        "make the indicator lamp shine",
        "I would like the board indicator lit", "can you turn the led on",
        "make sure the status led is on",
        "switch the indicator light on for me",
    ],
    "LED_OFF": [
        "please deactivate the status led",
        "make the red indicator turn off",
        "I need the green led extinguished", "disable the board indicator",
        "switch the blue status led into the off state",
        "the indicator light should go dark", "power down my led indicator",
        "set the status indicator as off", "turn my board led off please",
        "make the indicator lamp stop shining",
        "I would like the board indicator off", "can you turn the led off",
        "make sure the status led is off",
        "switch the indicator light off for me",
    ],
    "RGB_SET": [
        "choose cyan on the rgb module", "make the rgb output orange",
        "I want the rgb led to display violet",
        "select white for the rgb light", "change my rgb module to magenta",
        "set the rgb colour as green", "display pink using the rgb led",
        "make the rgb indicator yellow", "choose blue as the rgb colour",
        "change the rgb output to purple", "can you make the rgb led red",
        "switch the rgb light to blue",
        "I need the rgb module to show orange", "turn the rgb output white",
    ],
    "MATRIX_SHOW": [
        "put a question mark on the 8x8 panel",
        "draw the letter C on the matrix",
        "display a number 7 on the led matrix",
        "show a down arrow using the dot matrix",
        "render a smile on the 8x8 led display",
        "put a star symbol on the dot panel",
        "show a cross symbol using the led matrix",
        "display a happy face on the 8x8 panel",
        "draw a heart symbol on the led display",
        "show the letter Z on the matrix",
        "show a smiley face on the dot matrix",
        "put the number 8 on the 8x8 display",
        "display an up arrow on the led display",
        "draw a tick on the matrix",
    ],
    "MATRIX_CLEAR": [
        "remove the picture from the dot matrix",
        "make every matrix pixel turn dark",
        "erase the content on the 8x8 panel", "reset the led dot display",
        "get rid of the symbol on the matrix",
        "leave the led matrix completely blank",
        "wipe the image from the dot panel",
        "switch every pixel on the matrix off", "empty the 8x8 led display",
        "remove the current drawing from the matrix",
        "turn off the matrix completely", "clear the dot matrix screen",
        "take everything off the led panel",
        "reset the 8x8 display to blank",
    ],
    "MOTOR_MOVE": [
        "place the motor at 105 degrees", "target an angle of 195 degrees",
        "set the shaft position to 330 degrees",
        "move the stepper to 15 degrees",
        "rotate the motor until it reaches 255 degrees",
        "position the motor at 165 degrees",
        "set the stepper angle to 285 degrees",
        "move the shaft to 210 degrees", "take the motor to 345 degrees",
        "set the target position to 105 degrees",
        "move the stepper shaft to 150 degrees",
        "send the motor to 60 degrees", "turn the stepper to angle 90",
        "set the motor to 300 degrees",
    ],
    "MOTOR_CW": [
        "begin turning the stepper clockwise",
        "drive the shaft in a clockwise direction",
        "keep spinning in the clockwise direction",
        "activate clockwise movement",
        "make the motor rotate to the clockwise side",
        "start a clockwise spin", "the stepper should rotate clockwise",
        "continue the clockwise motor motion",
        "engage clockwise rotation on the shaft",
        "run the stepper in clockwise mode",
        "can you spin the stepper clockwise", "make the shaft go clockwise",
        "turn the motor clockwise please", "keep the motor going clockwise",
    ],
    "MOTOR_CCW": [
        "begin turning the stepper counterclockwise",
        "drive the shaft in a counterclockwise direction",
        "keep spinning in the counterclockwise direction",
        "activate counterclockwise movement",
        "make the motor rotate to the counterclockwise side",
        "start a counterclockwise spin",
        "the stepper should rotate counterclockwise",
        "continue the counterclockwise motor motion",
        "engage counterclockwise rotation on the shaft",
        "run the stepper in counterclockwise mode",
        "can you spin the stepper counter clockwise",
        "make the shaft go anticlockwise",
        "turn the motor anti clockwise please",
        "keep the motor going counterclockwise",
    ],
    "MOTOR_STOP": [
        "bring the stepper rotation to a halt",
        "make the shaft stop turning", "end all movement from the motor",
        "I need the stepper stationary",
        "stop the motor operation right away",
        "cancel whatever the stepper is doing",
        "halt the shaft movement immediately", "make the motor remain still",
        "stop all rotation now", "cease the stepper operation",
        "please stop the motor", "make the stepper come to a stop",
        "I want the shaft stationary", "abort the stepper movement",
    ],
    "BUZZER_PLAY": [
        "give me one quick beep", "sound a pair of beeps",
        "play the confirmation tone", "make the buzzer play an error signal",
        "trigger a repeating warning sound",
        "activate the steady active buzzer", "make a continuous alert tone",
        "play a rising success signal", "sound two quick beeps please",
        "play the failure signal", "make the alarm sound",
        "start a steady alert on the buzzer",
    ],
    "BUZZER_STOP": [
        "quiet the buzzer now", "end the beeping",
        "deactivate the buzzer alert", "stop the warning sound",
        "switch the active alert off", "please mute the buzzer",
        "turn off the alarm sound", "can you stop the beeping",
        "shut off the buzzer alarm", "make the buzzer be quiet",
    ],
}


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
    "BUZZER_PLAY",
    "BUZZER_STOP",
]

if list(training_data.keys()) != REQUIRED_CLASSES:
    missing = set(REQUIRED_CLASSES) - set(training_data.keys())
    extra = set(training_data.keys()) - set(REQUIRED_CLASSES)

    raise ValueError(
        f"Unexpected dataset classes. Missing={missing}, extra={extra}"
    )

for intent in REQUIRED_CLASSES:
    if intent not in test_data:
        raise ValueError(f"Missing test data for {intent}")


# ============================================================
# 3. UNKNOWN DATA
# ============================================================

# These teach the model that ordinary unrelated language, and commands
# for hardware this system does not support, are not supported actions.
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
    "turn on the porch light",
    "switch off the street light",
    "turn off the hall light",
    "turn on the living room light",
    "turn on the bathroom light",
    "switch off the garden lights",
    "switch off the standing lamp",
    "turn on the ceiling light",
    "stop the bus",
    "stop the engine",
    "stop the clock",
    "stop the video",
    "stop the recording",
    "stop the program",
    "stop the game",
    "stop the noise",
    "stop that",
    "cancel the flight",
    "cancel my appointment",
    "halt the production line",
    "stop the conveyor belt",
    "stop the fan",
    "switch off the air fan",
    "rotate the fan blades",
    "spin the wheel",
    "turn the wheel to the left",
    "stop the washing machine",
    "how does a stop sign work",
    "turn on the heater",
    "switch off the projector",
    "turn on the computer",
    "switch off the laptop",
    "turn on the printer",
    "turn off the speaker",
    "switch on the generator",
    "turn off the kettle",
    "turn on the table lamp",
    "turn off the flashlight",
    "switch off the water pump",
    "turn on the oven",
    "move the servo to 90 degrees",
    "rotate the robot arm",
    "play a tone on the speaker",
    "read the temperature sensor",
    "what is the temperature",
    "measure the distance",
    "scan for wifi networks",
    "connect to bluetooth",
    "start the camera",
    "take a photo",
    "display text on the lcd screen",
    "scroll text across the screen",
    "show the time on the display",
    "print hello world on the screen",
    "restart the board",
    "reboot the esp32",
    "open the garage door",
    "lock the door",
    "change the channel",
    "turn the knob",
    "turn the volume up",
    "paint the wall red",
    "make my website purple",
    "I like the colour blue",
    "what colour is the sky",
    "change the colour of my shirt",
    "what does clockwise mean",
    "explain clockwise rotation",
    "how fast can the motor spin",
    "what is a stepper driver",
    "the motor is overheating",
    "which direction does the motor rotate by default",
    "why is my led not working",
    "what colour led should I buy",
    "how do I wire an led",
    "is the led on",
    "is the indicator working",
    "what is the matrix for",
    "tell me about the matrix movie",
    "what is a matrix in mathematics",
    "multiply these two matrices",
    "stop the music",
    "stop the timer",
    "stop talking",
    "stop the download",
    "pause the video",
    "cancel my order",
    "cancel the meeting",
    "halt the process",
    "clear my browser history",
    "clear the table",
    "erase the whiteboard",
    "reset my password",
    "wipe the screen of my phone",
    "set the oven to 180 degrees",
    "rotate this image by 90 degrees",
    "what is ninety degrees in radians",
    "the angle of a triangle is 60 degrees",
    "turn left at the junction",
    "turn the page",
    "move the file to a folder",
    "set a timer for ten minutes",
    "set the temperature to twenty degrees",
    "move to Lagos next year",
    "set the table for dinner",
    "what does a buzzer do",
    "how do I wire a buzzer",
    "buy a new buzzer",
    "play my favourite playlist",
    "turn up the speaker volume",
    "mute the television",
    "silence my phone",
    "the alarm clock is broken",
    "set an alarm for six o clock",
    "wake me up with an alarm at seven",
    "what sound does a cat make",
    "sound the horn of the car",
    "play a video",
    "play a game",
    "make a sound like a dog",
    "play the radio station",
    "is the buzzer on",
    "are the lights on",
    "is the motor running",
    "is the matrix showing anything",
    "what is the led doing",
    "is the alarm on",
    "are the leds working",
    "how many leds are there",
]

unknown_test = [
    "i am called david",
    "my favourite subject is mathematics",
    "explain photosynthesis",
    "write a short story about a robot",
    "what is the capital of Ghana",
    "calculate the derivative of x squared",
    "I need to charge my phone",
    "please open the browser",
    "turn on the radio",
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
    "can you help with my homework",
    "what is the weather in Lagos",
    "how do I connect a keyboard",
    "show me a picture of a cat",
    "write a business plan",
    "what is the difference between AI and ML",
    "help me prepare for an examination",
    "turn on the bedroom light",
    "increase the television volume",
    "what is the meaning of success",
    "turn off the gas cooker",
    "switch on the standing fan",
    "turn on the water heater",
    "switch off the television please",
    "turn off the car engine",
    "move the robot forward",
    "rotate the camera to the left",
    "make the speaker louder",
    "show the temperature on the screen",
    "read the humidity sensor",
    "is the matrix finished",
    "what does rgb stand for",
    "how does a stepper motor spin",
    "what is counterclockwise",
    "paint the fence green",
    "I like the colour orange",
    "stop the car",
    "stop the stopwatch",
    "cancel the subscription",
    "clear the whiteboard",
    "erase my history",
    "reset the router",
    "set the thermostat to twenty five degrees",
    "turn the photo ninety degrees",
    "I need a new phone",
    "who won the match yesterday",
    "how far is Abuja from Lagos",
    "describe the water cycle",
    "give me a recipe for jollof rice",
    "what is a resistor",
    "how loud is a buzzer",
    "turn the volume down",
    "mute my microphone",
    "stop the ringing of my phone",
    "play the news",
    "make a sound like a cat",
    "set the alarm for tomorrow morning",
    "play the next episode",
]


# ============================================================
# 4. TEXT PROCESSING
# ============================================================

# This exact tokenizer + vocabulary + multi-hot encoding must be
# reproduced in the ESP32 firmware.

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
                f"Conflicting training phrase: {phrase!r} appears in "
                f"{phrase_owner[key]} and {intent}"
            )

        phrase_owner[key] = intent


# A supported phrase must not also be taught as UNKNOWN.
for phrase in unknown_training:
    key = normalize(phrase)

    if key in phrase_owner:
        raise ValueError(
            f"Phrase is both {phrase_owner[key]} and UNKNOWN: {phrase!r}"
        )


# Remove duplicate phrases within the same class.
for intent in REQUIRED_CLASSES:
    unique = {}
    for phrase in training_data[intent]:
        unique[normalize(phrase)] = phrase
    training_data[intent] = list(unique.values())

unknown_training = list(
    {normalize(p): p for p in unknown_training}.values()
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

layers = [
    tf.keras.layers.Input(
        shape=(len(vocabulary),),
        name="word_presence"
    ),
    tf.keras.layers.Dense(HIDDEN_UNITS[0], activation="relu"),
]

if DROPOUT_RATE > 0:
    layers.append(tf.keras.layers.Dropout(DROPOUT_RATE))

layers.append(tf.keras.layers.Dense(HIDDEN_UNITS[1], activation="relu"))

layers.append(
    tf.keras.layers.Dense(
        len(classes),
        activation="softmax",
        name="intent_scores"
    )
)

model = tf.keras.Sequential(layers)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=0.00005
    ),
]


print("\n====================================")
print("AIEng4D INTENT MODEL")
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
    epochs=250,
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)


# ============================================================
# 6. PARAMETER EXTRACTION AND DEVICE VALIDATION
# ============================================================

# The neural network only suggests an intent. These deterministic rules
# decide whether the command is complete, unambiguous and aimed at the
# right device. Anything doubtful is rejected: a rejected command is
# always safer than a wrong one.
#
# RGB colour aliases. The RGB values are sent to the firmware as
# parameters. Physical RGB wiring (common anode / common cathode) must be
# checked when implementing the hardware controller.

RGB_COLOURS = {
    "red": {"aliases": ["red"], "rgb": [255, 0, 0]},
    "green": {"aliases": ["green"], "rgb": [0, 255, 0]},
    "blue": {"aliases": ["blue"], "rgb": [0, 0, 255]},
    "yellow": {"aliases": ["yellow"], "rgb": [255, 255, 0]},
    "cyan": {"aliases": ["cyan", "aqua"], "rgb": [0, 255, 255]},
    "white": {"aliases": ["white"], "rgb": [255, 255, 255]},
    "orange": {"aliases": ["orange"], "rgb": [255, 128, 0]},
    "pink": {"aliases": ["pink"], "rgb": [255, 105, 180]},
    "magenta": {"aliases": ["magenta"], "rgb": [255, 0, 255]},
    "purple": {"aliases": ["purple", "violet"], "rgb": [128, 0, 255]},
}


# Symbols the matrix firmware can draw. This table is the contract with
# the firmware: do not add a symbol here until the glyph exists there.
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
    "number_3": ["number 3", "number three"],
    "number_5": ["number 5", "number five"],
    "number_7": ["number 7", "number seven"],
    "number_8": ["number 8", "number eight"],
}


# Words that name each device. A command must name the device it targets.
LED_WORDS = [
    "led", "indicator", "status light", "board led",
    "indicator light", "status indicator",
]

MOTOR_WORDS = ["motor", "stepper", "shaft"]

MATRIX_WORDS = [
    "matrix", "8x8", "8 by 8", "dot display", "dot panel",
    "led panel", "led display", "symbol display",
]

# Every word seen in the motor direction / stop training phrases. Used to
# decide whether a command that never names the motor is still a motor
# command.
MOTOR_COMMAND_VOCABULARY = {
    word
    for intent in ("MOTOR_CW", "MOTOR_CCW", "MOTOR_STOP")
    for phrase in training_data[intent]
    for word in tokenize(phrase)
}

# In this system only the stepper motor rotates, so a rotation word counts
# as naming the motor. It does NOT count if another actuator is named.
ROTATION_WORDS = [
    "rotate", "rotates", "rotating", "rotation",
    "spin", "spins", "spinning",
]

UNSUPPORTED_ROTATING_THINGS = [
    "servo", "fan", "wheel", "wheels", "robot", "arm", "camera", "door",
    "car", "window", "knob", "dial", "earth", "planet", "propeller",
    "drill", "engine", "turbine", "pump", "conveyor", "blade", "blades",
]

# If any of these appear, the command is not about the single indicator LED.
OTHER_DEVICE_WORDS = [
    "rgb", "matrix", "8x8", "8 by 8", "panel", "display", "dot",
    "screen", "motor", "stepper", "shaft",
]

LED_ON_WORDS = [
    "on", "turn on", "switch on", "activate", "activated", "enable",
    "enabled", "illuminate", "illuminated", "glow", "glowing", "shine",
    "shining", "light up", "lit", "power on", "power up", "come on",
    "start", "visible", "light the",
]

LED_OFF_WORDS = [
    "off", "turn off", "switch off", "deactivate", "deactivated",
    "disable", "disabled", "extinguish", "extinguished", "dark",
    "go dark", "power down", "shut down", "shut off", "stop glowing",
    "stop shining", "stop the led", "stop the indicator",
    "stop the status", "from glowing", "from shining", "put out", "kill",
]

# Words that contradict the requested LED state.
LED_ON_CONTRADICTIONS = [
    "off", "switch off", "turn off", "deactivate", "deactivated",
    "disable", "disabled", "extinguish", "extinguished", "dark",
    "power down", "shut down", "shut off",
]

LED_OFF_CONTRADICTIONS = [
    "turn on", "switch on", "activate", "enable", "glow", "illuminate",
    "light up", "power on", "come on",
]

CLOCKWISE_ALIASES = ["clockwise", "clock wise"]

COUNTERCLOCKWISE_ALIASES = [
    "counterclockwise", "counter clockwise", "counter clock wise",
    "anticlockwise", "anti clockwise", "anti clock wise",
]

# A stop request is always the safe state, so this list is deliberately
# generous. A stop accepted by mistake is harmless; a missed stop is not.
MOTOR_STOP_WORDS = [
    "stop", "stopped", "stopping", "halt", "halted", "pause", "cancel",
    "cease", "stationary", "still", "off", "abort", "terminate",
    "freeze", "end", "disable", "rest", "standstill",
]

# "do not move/rotate/turn the motor" is a stop request.
STOP_NEGATED_ACTION = re.compile(
    r"\b(?:do not|don't|dont|never)\s+(?:move|rotate|turn|spin|run)\b"
)

# CW / CCW mean "keep rotating in this direction". Any amount ("90
# degrees", "twice", "a quarter turn") is a different, relative command
# that is not supported, so it must be rejected rather than turned into
# an endless rotation.
ROTATION_AMOUNT_WORDS = [
    "degree", "degrees", "deg", "once", "twice", "thrice", "times",
    "revolution", "revolutions", "full", "half", "quarter", "step",
    "steps", "turns", "rotations",
]

# Words that make an angle relative or mixed with another action.
# Only absolute target angles are supported.
MOTOR_MOVE_REJECT_WORDS = [
    "by", "more", "another", "further", "additional", "extra", "back",
    "again", "plus", "minus", "relative",
    "stop", "halt", "pause", "cancel", "abort", "freeze",
]

MATRIX_CLEAR_WORDS = [
    "clear", "blank", "erase", "remove", "reset", "wipe", "empty",
    "dark", "off", "hide", "delete", "stop", "get rid of",
    "take everything off", "clean",
]

# Words that mean the user wants to draw something.
MATRIX_DRAW_WORDS = ["show", "display", "draw", "put", "render", "write"]


# Fixed sound patterns implemented in firmware. The model never generates
# arbitrary audio; it only selects one of these names. The pattern names
# and the passive / active device split are the contract with the firmware.
BUZZER_PATTERNS = {
    "SHORT_BEEP": {
        "aliases": [
            "beep", "short beep", "single beep", "one beep",
            "short tone", "single tone", "quick beep",
        ],
        "device": "passive",
        "description": "One short tone",
    },
    "DOUBLE_BEEP": {
        "aliases": [
            "double beep", "two beeps", "two short beeps", "double tone",
            "pair of beeps", "two quick beeps",
        ],
        "device": "passive",
        "description": "Two short tones",
    },
    "SUCCESS": {
        "aliases": [
            "success", "success sound", "success tone",
            "confirmation sound", "confirmation tone", "success signal",
            "ascending tone", "rising success signal",
        ],
        "device": "passive",
        "description": "Three ascending tones",
    },
    "ERROR": {
        "aliases": [
            "error", "error sound", "error tone", "error signal",
            "failure sound", "failure signal", "failure tone",
            "low error beep",
        ],
        "device": "passive",
        "description": "Two low tones",
    },
    "ALARM": {
        "aliases": [
            "alarm", "alarm sound", "alarm pattern", "warning alarm",
            "warning sound", "repeating warning sound",
        ],
        "device": "passive",
        "description": "Repeating alternating tones",
    },
    "ACTIVE_ALERT_ON": {
        "aliases": [
            "active buzzer", "continuous alert", "continuous buzzer",
            "steady alert", "active alert", "steady active buzzer",
            "continuous buzzer alert",
        ],
        "device": "active",
        "description": "Continuous active-buzzer alert",
    },
}

BUZZER_STOP_ACTION = {
    "action": "stop_all",
    "description": "Stop both passive and active buzzers",
}

BUZZER_SOUND_WORDS = [
    "buzzer", "sound", "beep", "beeps", "beeping", "tone", "alarm",
    "alert", "signal",
]

BUZZER_PLAY_CONTRADICTIONS = [
    "stop", "silence", "quiet", "off", "deactivate", "disable", "halt",
    "cancel", "mute", "shut off",
]

BUZZER_STOP_WORDS = [
    "stop", "silence", "quiet", "off", "deactivate", "disable", "halt",
    "cancel", "end", "mute", "shut off",
]

# A stop request must not also ask for a new sound.
BUZZER_STOP_CONTRADICTIONS = [
    "play", "start", "activate", "trigger", "turn on", "switch on",
    "enable", "begin", "produce", "give",
]

# Sound-related things that are NOT the buzzer.
UNSUPPORTED_SOUND_THINGS = [
    "music", "song", "songs", "playlist", "radio", "speaker", "speakers",
    "video", "podcast", "voice", "phone", "television", "tv",
    "microphone", "game", "clock", "timer", "stopwatch", "horn",
    "ringtone", "news", "episode",
]

# An alarm with a time or date is an alarm-clock request, not a buzzer
# pattern.
BUZZER_NON_COMMAND_WORDS = [
    "set", "tomorrow", "morning", "tonight", "minutes", "hours", "oclock",
]


# A command that starts with one of these is a question about state.
QUESTION_WORDS = [
    "is", "are", "was", "were", "why", "what", "whats", "how", "when",
    "where", "which", "who", "whose", "whom", "does", "did",
]


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
        r"\bdont\b",
        r"\bdo not\b",
        r"\bdoes not\b",
        r"\bshouldn't\b",
        r"\bshould not\b",
        r"\bcan't\b",
        r"\bcannot\b",
    ]

    lowered = text.lower()

    return any(re.search(p, lowered) for p in patterns)


def direction_flags(text):
    """
    Return (has_clockwise, has_counterclockwise).

    Counterclockwise phrases are removed before looking for "clockwise",
    because "counter clockwise" contains the word "clockwise". Without
    this, a counterclockwise command would be read as clockwise.
    """

    stripped = normalize(text)
    has_counter = False

    for alias in sorted(
        COUNTERCLOCKWISE_ALIASES, key=len, reverse=True
    ):
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalize(alias))
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, stripped):
            has_counter = True
            stripped = re.sub(pattern, " ", stripped)

    has_clockwise = contains_any(stripped, CLOCKWISE_ALIASES)

    return has_clockwise, has_counter


def names_motor(lowered):
    """
    True if the command clearly targets the stepper motor.

    The motor must be named, or a rotation word must be used with only
    words seen in motor commands. Naming any other actuator (servo, fan, wheel...)
    always means "not our motor".
    """

    if contains_any(lowered, UNSUPPORTED_ROTATING_THINGS):
        return False

    if contains_any(lowered, MOTOR_WORDS):
        return True

    # No motor named: a rotation word is enough only if EVERY word has
    # appeared in a real motor command. A word from elsewhere ("rotate
    # the photo clockwise") means the command is about something else.
    return (
        contains_any(lowered, ROTATION_WORDS)
        and all(
            token in MOTOR_COMMAND_VOCABULARY
            for token in tokenize(lowered)
        )
    )


def extract_motor_angle(text):
    """
    Extract exactly one absolute angle.

    Accepted examples:
        move motor to 90 degrees
        set motor angle to 45 degrees
        move stepper to angle 120

    Rejects:
        move motor somewhere
        move motor to 900 degrees
        move motor to 90 and 180 degrees
        rotate the motor by 90 degrees          (relative)
        rotate the motor 90 degrees clockwise   (relative)
        stop the motor and move to 90 degrees   (mixed)
    """

    lowered = text.lower()

    has_cw, has_ccw = direction_flags(lowered)

    if has_cw or has_ccw:
        return None

    if contains_any(lowered, MOTOR_MOVE_REJECT_WORDS):
        return None

    numbers = re.findall(
        r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\d.])",
        lowered
    )

    if len(set(numbers)) != 1 or len(numbers) != 1:
        return None

    if not contains_any(
        lowered,
        ["degree", "degrees", "deg", "angle", "position", "target"]
    ):
        return None

    angle = float(numbers[0])

    if angle < MOTOR_MIN_DEGREES:
        return None

    if angle > MOTOR_MAX_DEGREES:
        return None

    if not angle.is_integer():
        return None

    return int(angle)


def extract_matrix_symbol(text):
    matches = find_matches(text, MATRIX_SYMBOLS)

    # Only accept exactly one recognized symbol.
    if len(matches) != 1:
        return None

    return matches[0]


def extract_buzzer_pattern(text):
    """
    Return exactly one buzzer pattern, preferring the most specific
    (longest) alias. Two equally specific patterns mean the command is
    ambiguous, so nothing is returned.
    """

    lowered = normalize(text)
    matches = []

    for canonical, details in BUZZER_PATTERNS.items():
        for alias in details["aliases"]:
            normalized_alias = normalize(alias)

            if contains_any(lowered, [normalized_alias]):
                matches.append(
                    (canonical, len(normalized_alias.split()))
                )

    if not matches:
        return None

    longest = max(length for _, length in matches)

    best = {
        canonical for canonical, length in matches
        if length == longest
    }

    if len(best) != 1:
        return None

    return best.pop()


def parse_parameters(intent, text):
    """
    Return a parameter dictionary only when the command is complete,
    unambiguous and aimed at the right device. Otherwise return None.
    """

    text = str(text).replace("\u2019", "'")
    lowered = normalize(text)

    # A question is not a command: "is the led on" must never turn it on.
    first_words = tokenize(text)[:1]

    if first_words and first_words[0] in QUESTION_WORDS:
        return None

    if has_negation(text):
        # The only negated command accepted is "do not move the motor",
        # which is a stop request.
        if not (
            intent == "MOTOR_STOP"
            and STOP_NEGATED_ACTION.search(text.lower())
        ):
            return None

    if intent == "LED_ON":
        if contains_any(lowered, OTHER_DEVICE_WORDS):
            return None

        if not contains_any(lowered, LED_WORDS):
            return None

        if contains_any(lowered, LED_ON_CONTRADICTIONS):
            return None

        if not contains_any(lowered, LED_ON_WORDS):
            return None

        return {}

    if intent == "LED_OFF":
        if contains_any(lowered, OTHER_DEVICE_WORDS):
            return None

        if not contains_any(lowered, LED_WORDS):
            return None

        if contains_any(lowered, LED_OFF_CONTRADICTIONS):
            return None

        if not contains_any(lowered, LED_OFF_WORDS):
            return None

        return {}

    if intent == "RGB_SET":
        if not contains_any(lowered, ["rgb"]):
            return None

        if contains_any(lowered, MATRIX_WORDS + MOTOR_WORDS):
            return None

        matches = find_matches(lowered, {
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
        # A target angle moves hardware, so the motor must be NAMED.
        # A rotation word alone is not enough here.
        if contains_any(lowered, UNSUPPORTED_ROTATING_THINGS):
            return None

        if not contains_any(lowered, MOTOR_WORDS):
            return None

        angle = extract_motor_angle(text)

        if angle is None:
            return None

        return {
            "angle_degrees": angle
        }

    if intent in ("MOTOR_CW", "MOTOR_CCW"):
        if not names_motor(lowered):
            return None

        # "rotate 90 degrees clockwise" is a relative move, not a
        # continuous rotation.
        if re.search(r"\d", lowered) or contains_any(
            lowered, ROTATION_AMOUNT_WORDS
        ):
            return None

        has_cw, has_ccw = direction_flags(lowered)

        # Both directions in one command is ambiguous.
        if has_cw and has_ccw:
            return None

        if intent == "MOTOR_CW" and has_cw:
            return {"direction": "clockwise"}

        if intent == "MOTOR_CCW" and has_ccw:
            return {"direction": "counterclockwise"}

        return None

    if intent == "MOTOR_STOP":
        if not names_motor(lowered):
            return None

        # A number means a target or a second motor: not a plain stop.
        if re.search(r"\d", lowered):
            return None

        negated_move = bool(
            STOP_NEGATED_ACTION.search(text.lower())
        )

        if not negated_move and not contains_any(
            lowered, MOTOR_STOP_WORDS
        ):
            return None

        return {"action": "stop"}

    if intent == "MATRIX_SHOW":
        if not contains_any(lowered, MATRIX_WORDS):
            return None

        # "show a heart and clear the matrix" is two commands.
        if contains_any(
            lowered, ["clear", "erase", "blank", "remove", "wipe"]
        ):
            return None

        symbol = extract_matrix_symbol(lowered)

        if symbol is None:
            return None

        return {
            "symbol": symbol
        }

    if intent in ("BUZZER_PLAY", "BUZZER_STOP"):
        # Another device, or a sound source we do not control.
        if contains_any(
            lowered,
            OTHER_DEVICE_WORDS + ["led", "indicator"]
            + UNSUPPORTED_SOUND_THINGS
        ):
            return None

        # "set an alarm for six o clock" is an alarm-clock request.
        if re.search(r"\d", lowered) or contains_any(
            lowered, BUZZER_NON_COMMAND_WORDS
        ):
            return None

        if not contains_any(lowered, BUZZER_SOUND_WORDS):
            return None

        if intent == "BUZZER_PLAY":
            # A play request cannot also contain stop language.
            if contains_any(lowered, BUZZER_PLAY_CONTRADICTIONS):
                return None

            pattern = extract_buzzer_pattern(lowered)

            if pattern is None:
                return None

            return {
                "device": BUZZER_PATTERNS[pattern]["device"],
                "pattern": pattern,
            }

        if not contains_any(lowered, BUZZER_STOP_WORDS):
            return None

        if contains_any(lowered, BUZZER_STOP_CONTRADICTIONS):
            return None

        return {"action": BUZZER_STOP_ACTION["action"]}

    if intent == "MATRIX_CLEAR":
        if not contains_any(lowered, MATRIX_WORDS):
            return None

        if not contains_any(lowered, MATRIX_CLEAR_WORDS):
            return None

        # "clear the matrix and show a heart" is two commands.
        if (
            extract_matrix_symbol(lowered) is not None
            and contains_any(lowered, MATRIX_DRAW_WORDS)
        ):
            return None

        return {
            "action": "clear"
        }

    return None


# ============================================================
# 7. COMMAND CLASSIFICATION AND REJECTION
# ============================================================

def decide(text, scores):
    """
    Apply every rejection rule to a text and its class scores.

    A command is accepted only if:
      1. It is not empty.
      2. It has adequate vocabulary coverage.
      3. The classifier predicts a supported intent.
      4. Confidence passes the initial threshold.
      5. The top-two score margin passes the threshold.
      6. Deterministic parameter extraction and device checks succeed.

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

    parameters = parse_parameters(predicted_intent, text)

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


def classify_command(text):
    """Full command pipeline for one text."""

    if not isinstance(text, str) or not text.strip():
        return decide(text, None)

    vector = encode_text(text).reshape(1, -1)
    scores = model.predict(vector, verbose=0)[0]

    return decide(text, scores)


def short_reason(result):
    if result["accepted"]:
        return result["intent"]

    return result["reason"]


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


# Hard guard: nothing in the held-out set may also be a training phrase.
# A leaked phrase silently inflates every reported score.
train_keys = {normalize(t) for t in train_texts}

leaked_held_out = sorted({
    t for t in test_texts if normalize(t) in train_keys
})

if leaked_held_out:
    raise ValueError(
        "Held-out phrases also appear in training: "
        + ", ".join(repr(t) for t in leaked_held_out)
    )

if len({normalize(t) for t in test_texts}) != len(test_texts):
    raise ValueError("Duplicate phrases inside the held-out set.")


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

misclassified_lines = []
prediction_lines = []

for text, actual, predicted, scores in zip(
    test_texts, y_test, test_predictions, test_scores
):
    line = (
        f"{text:62s} | actual={classes[actual]:12s} | "
        f"predicted={classes[predicted]:12s} | "
        f"confidence={float(scores[predicted]):.2%}"
    )

    prediction_lines.append(line)

    if actual != predicted:
        misclassified_lines.append(line)


# ============================================================
# 9. END-TO-END HELD-OUT TEST (classifier + all rejection rules)
# ============================================================

# This is the metric that matters. The raw classifier score above says how
# well the network labels text; this says what the full system would DO.
#
#   ACCEPTED_OK            supported command accepted with the right intent
#   REJECTED_SUPPORTED     supported command rejected (annoying, but safe)
#   WRONG_INTENT_ACCEPTED  accepted as the wrong action   (DANGEROUS)
#   UNKNOWN_REJECTED       unsupported text rejected       (correct)
#   FALSE_ACCEPT           unsupported text accepted       (DANGEROUS)

e2e_counts = {
    "ACCEPTED_OK": 0,
    "REJECTED_SUPPORTED": 0,
    "WRONG_INTENT_ACCEPTED": 0,
    "UNKNOWN_REJECTED": 0,
    "FALSE_ACCEPT": 0,
}

e2e_problem_lines = []
e2e_per_class = {name: {"ok": 0, "total": 0} for name in classes}

for text, label, scores in zip(test_texts, y_test, test_scores):
    expected = classes[label]
    result = decide(text, scores)

    e2e_per_class[expected]["total"] += 1

    if expected == UNKNOWN_CLASS:
        if result["accepted"]:
            outcome = "FALSE_ACCEPT"
        else:
            outcome = "UNKNOWN_REJECTED"
    else:
        if result["accepted"] and result["intent"] == expected:
            outcome = "ACCEPTED_OK"
        elif result["accepted"]:
            outcome = "WRONG_INTENT_ACCEPTED"
        else:
            outcome = "REJECTED_SUPPORTED"

    e2e_counts[outcome] += 1

    if outcome in ("ACCEPTED_OK", "UNKNOWN_REJECTED"):
        e2e_per_class[expected]["ok"] += 1
    else:
        e2e_problem_lines.append(
            f"{outcome:22s} | {text:58s} | expected={expected:12s} | "
            f"{short_reason(result)}"
        )

e2e_supported_total = len(test_texts) - len(unknown_test)
e2e_unknown_total = len(unknown_test)


# ============================================================
# 10. PARSER COVERAGE (ground-truth intent, no classifier involved)
# ============================================================

# For every dataset phrase: does the deterministic parser accept it when
# told the correct intent? A phrase the network is trained to recognise
# but the parser always refuses is a gap between the two halves.

parser_coverage_lines = []
parser_coverage = {}

for split_name, split_data in (
    ("training", training_data),
    ("held-out", test_data),
):
    for intent in REQUIRED_CLASSES:
        phrases = split_data[intent]
        failed = [
            p for p in phrases
            if parse_parameters(intent, p) is None
        ]

        parser_coverage[(split_name, intent)] = (
            len(phrases) - len(failed), len(phrases)
        )

        for phrase in failed:
            parser_coverage_lines.append(
                f"{split_name:9s} | {intent:12s} | {phrase}"
            )


# ============================================================
# 11. CURATED COMMAND PIPELINE TESTS
# ============================================================

# Each test is (text, expected_intent, expected_accepted, expected_params).
# None for expected_params means "do not check parameters".
# None for expected_intent means the command must be rejected.
# None of these phrases may appear in the training data (checked below).
# The "hazard" group are commands that would have caused a wrong or
# dangerous action in an earlier parser.

pipeline_tests = [
    # --- supported commands ---
    ("Please turn the status LED on", "LED_ON", True, {}),
    ("TURN ON THE STATUS LED!!!", "LED_ON", True, {}),
    ("Shut off the board LED", "LED_OFF", True, {}),
    ("Set the RGB LED to purple", "RGB_SET", True,
     {"colour": "purple", "rgb": [128, 0, 255]}),
    ("Make the RGB light cyan", "RGB_SET", True,
     {"colour": "cyan", "rgb": [0, 255, 255]}),
    ("Change the RGB module to orange", "RGB_SET", True,
     {"colour": "orange", "rgb": [255, 128, 0]}),
    ("Move the stepper motor to 75 degrees", "MOTOR_MOVE", True,
     {"angle_degrees": 75}),
    ("Bring the motor shaft to 225 degrees", "MOTOR_MOVE", True,
     {"angle_degrees": 225}),
    ("Rotate the stepper clockwise", "MOTOR_CW", True,
     {"direction": "clockwise"}),
    ("Rotate the motor counterclockwise", "MOTOR_CCW", True,
     {"direction": "counterclockwise"}),
    ("Halt the stepper motor right now", "MOTOR_STOP", True,
     {"action": "stop"}),
    ("Please do not move the motor", "MOTOR_STOP", True,
     {"action": "stop"}),
    ("Show a star on the LED matrix", "MATRIX_SHOW", True,
     {"symbol": "star"}),
    ("Show the letter A on the matrix", "MATRIX_SHOW", True,
     {"symbol": "letter_a"}),
    ("Wipe the LED matrix", "MATRIX_CLEAR", True,
     {"action": "clear"}),
    ("Switch off the LED matrix", "MATRIX_CLEAR", True,
     {"action": "clear"}),

    # --- direction hazards (counterclockwise must never become clockwise) ---
    ("Make the shaft rotate counter clockwise", "MOTOR_CCW", True,
     {"direction": "counterclockwise"}),
    ("Spin the stepper anticlockwise", "MOTOR_CCW", True,
     {"direction": "counterclockwise"}),
    ("Turn the shaft anti clockwise", "MOTOR_CCW", True,
     {"direction": "counterclockwise"}),
    ("Turn the motor clockwise and counterclockwise", None, False, None),

    # --- buzzer ---
    ("Please make a quick beep", "BUZZER_PLAY", True,
     {"device": "passive", "pattern": "SHORT_BEEP"}),
    ("Play a double beep for me", "BUZZER_PLAY", True,
     {"device": "passive", "pattern": "DOUBLE_BEEP"}),
    ("Could you play the success tone", "BUZZER_PLAY", True,
     {"device": "passive", "pattern": "SUCCESS"}),
    ("Sound the alarm now", "BUZZER_PLAY", True,
     {"device": "passive", "pattern": "ALARM"}),
    ("Switch on the continuous buzzer", "BUZZER_PLAY", True,
     {"device": "active", "pattern": "ACTIVE_ALERT_ON"}),
    ("Please silence the buzzer", "BUZZER_STOP", True,
     {"action": "stop_all"}),
    ("Stop the beeping now", "BUZZER_STOP", True,
     {"action": "stop_all"}),
    ("Switch off the alarm sound", "BUZZER_STOP", True,
     {"action": "stop_all"}),
    ("Play a sound", None, False, None),
    ("Play a beep and an alarm", None, False, None),
    ("Stop the buzzer and play a success sound", None, False, None),
    ("Do not beep", None, False, None),
    ("Turn on the active buzzer and stop it", None, False, None),
    ("Activate the buzzer", None, False, None),
    ("Set an alarm for seven", None, False, None),
    ("Play my favourite song", None, False, None),
    ("Is the buzzer sounding", None, False, None),
    ("Are the indicator lights lit", None, False, None),
    ("Make the motor beep", None, False, None),
    ("Turn on the LED and sound the alarm", None, False, None),
    ("Stop the radio", None, False, None),

    # --- rotation context: only the stepper rotates in this system ---
    ("Begin a counterclockwise spin", "MOTOR_CCW", True,
     {"direction": "counterclockwise"}),
    ("Please stop all the rotation", "MOTOR_STOP", True,
     {"action": "stop"}),
    ("Spin the fan clockwise", None, False, None),
    ("Rotate the picture clockwise", None, False, None),
    ("Rotate the photo anticlockwise", None, False, None),
    ("Stop the fan motor", None, False, None),
    ("Spin the servo clockwise", None, False, None),
    ("Stop the rotation of the earth", None, False, None),
    ("Move the servo motor to 90 degrees", None, False, None),

    # --- wrong-device hazards ---
    ("Turn off the RGB LED", None, False, None),
    ("Turn the RGB LED off", None, False, None),
    ("Switch on the LED matrix", None, False, None),
    ("Set the LED to red", None, False, None),

    # --- relative, mixed or out-of-range motor commands ---
    ("Rotate the motor 90 degrees clockwise", None, False, None),
    ("Rotate the stepper clockwise twice", None, False, None),
    ("Turn the motor a quarter turn counterclockwise", None, False, None),
    ("Move the stepper by 90 degrees", None, False, None),
    ("Stop the motor and move it to 90 degrees", None, False, None),
    ("Move the motor somewhere", None, False, None),
    ("Move the motor to 900 degrees", None, False, None),
    ("Move the motor to -30 degrees", None, False, None),
    ("Move the motor to 45.5 degrees", None, False, None),
    ("Move the motor to 90 and 180 degrees", None, False, None),

    # --- ambiguous or incomplete commands ---
    ("Make it purple", None, False, None),
    ("Set RGB to purple and orange", None, False, None),
    ("Turn on the LED and turn it off", None, False, None),
    ("Do not turn on the LED", None, False, None),
    ("Show something on the matrix", None, False, None),
    ("Show an unsupported symbol on the matrix", None, False, None),
    ("Show a heart and a star on the matrix", None, False, None),
    ("Show a heart and clear the matrix", None, False, None),

    # --- questions and unsupported requests ---
    ("Is the indicator currently on", None, False, None),
    ("Why does my LED keep failing", None, False, None),
    ("What is the meaning of clockwise", None, False, None),
    ("Stop the playlist", None, False, None),
    ("Turn the servo to 45 degrees", None, False, None),
    ("Turn off the air conditioner", None, False, None),
    ("I am called David", None, False, None),
    ("Recite a nursery rhyme", None, False, None),
    ("Calculate nine minus four", None, False, None),

    # --- empty and gibberish ---
    ("", None, False, None),
    ("     ", None, False, None),
    ("asdf qwerty zxcv", None, False, None),
]

leaked_pipeline = [
    p for p, _, _, _ in pipeline_tests
    if p.strip() and normalize(p) in train_keys
]

if leaked_pipeline:
    raise ValueError(
        "Pipeline test phrases also appear in training: "
        + ", ".join(repr(p) for p in leaked_pipeline)
    )

pipeline_lines = []
pipeline_failures = []
pipeline_passed = 0

print("\n====================================")
print("COMMAND PIPELINE TESTS")
print("====================================")

for phrase, expected_intent, expected_accepted, expected_params in (
    pipeline_tests
):
    result = classify_command(phrase)

    actual_accepted = result["accepted"]

    if expected_accepted:
        passed = (
            actual_accepted
            and result.get("intent") == expected_intent
            and (
                expected_params is None
                or result.get("parameters") == expected_params
            )
        )
    else:
        passed = not actual_accepted

    if passed:
        pipeline_passed += 1
    else:
        pipeline_failures.append(phrase)

    status = "PASS" if passed else "FAIL"

    detail = short_reason(result)

    if actual_accepted:
        detail += f" {result['parameters']}"

    line = f"{status:4s} | {phrase!r:58s} | {detail}"

    print(line)
    pipeline_lines.append(line)

pipeline_accuracy = (
    pipeline_passed / len(pipeline_tests)
) * 100


# ============================================================
# 12. KERAS SAVE AND TFLITE INT8 EXPORT
# ============================================================

model_path = OUTPUT_DIR / "intent_model.keras"
tflite_path = OUTPUT_DIR / "intent_model_int8.tflite"

model.save(model_path)


def representative_dataset():
    # Training examples from all classes, in the same multi-hot format
    # that the firmware will feed to the model.
    for index in range(len(X_train)):
        yield [X_train[index:index + 1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = None
tflite_info = {}
quantization_status = ""
tflite_test_accuracy = None
tflite_keras_agreement = None
tflite_pipeline_agreement = None

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


def tflite_scores(tflite_bytes, vectors):
    """Run the INT8 model exactly as firmware would."""

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]

    all_scores = []

    for vector in vectors:
        quantized = np.clip(
            np.round(vector / input_scale + input_zero),
            -128,
            127
        ).astype(np.int8).reshape(1, -1)

        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()

        raw = interpreter.get_tensor(output_detail["index"])[0]

        all_scores.append(
            (raw.astype(np.float32) - output_zero) * output_scale
        )

    info = {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero),
        "input_shape": [int(v) for v in input_detail["shape"]],
        "output_shape": [int(v) for v in output_detail["shape"]],
    }

    return np.asarray(all_scores), info


if tflite_model is not None:
    quantized_scores, tflite_info = tflite_scores(tflite_model, X_test)
    quantized_predictions = np.argmax(quantized_scores, axis=1)

    tflite_test_accuracy = float(
        np.mean(quantized_predictions == y_test)
    )

    tflite_keras_agreement = float(
        np.mean(quantized_predictions == test_predictions)
    )

    # Would the full system make the same accept/reject decision?
    same_decision = 0

    for text, keras_scores, int8_scores in zip(
        test_texts, test_scores, quantized_scores
    ):
        a = decide(text, keras_scores)
        b = decide(text, int8_scores)

        if (
            a["accepted"] == b["accepted"]
            and a.get("intent") == b.get("intent")
        ):
            same_decision += 1

    tflite_pipeline_agreement = same_decision / len(test_texts)


# ============================================================
# 13. SAVE VOCABULARY, CLASSES, PARSER CONFIG AND METADATA
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
    "buzzer_patterns": BUZZER_PATTERNS,
    "buzzer_stop_action": BUZZER_STOP_ACTION,
    "unsupported_sound_things": UNSUPPORTED_SOUND_THINGS,
    "question_words": QUESTION_WORDS,
    "matrix_symbols": MATRIX_SYMBOLS,
    "motor_min_degrees": MOTOR_MIN_DEGREES,
    "motor_max_degrees": MOTOR_MAX_DEGREES,
    "led_words": LED_WORDS,
    "motor_words": MOTOR_WORDS,
    "rotation_words": ROTATION_WORDS,
    "motor_command_vocabulary": sorted(MOTOR_COMMAND_VOCABULARY),
    "unsupported_rotating_things": UNSUPPORTED_ROTATING_THINGS,
    "matrix_words": MATRIX_WORDS,
    "other_device_words": OTHER_DEVICE_WORDS,
    "led_on_words": LED_ON_WORDS,
    "led_off_words": LED_OFF_WORDS,
    "led_on_contradictions": LED_ON_CONTRADICTIONS,
    "led_off_contradictions": LED_OFF_CONTRADICTIONS,
    "clockwise_aliases": CLOCKWISE_ALIASES,
    "counterclockwise_aliases": COUNTERCLOCKWISE_ALIASES,
    "motor_stop_words": MOTOR_STOP_WORDS,
    "motor_move_reject_words": MOTOR_MOVE_REJECT_WORDS,
    "rotation_amount_words": ROTATION_AMOUNT_WORDS,
    "matrix_clear_words": MATRIX_CLEAR_WORDS,
    "matrix_draw_words": MATRIX_DRAW_WORDS,
    "requires_explicit_device_context": True,
    "requires_valid_parameters": True,
    "negation_rejection": True,
    "negated_motion_means_stop": True,
    "multiple_parameter_rejection": True,
    "relative_angle_rejection": True,
    "direction_check_strips_counterclockwise_first": True,
}

parser_path.write_text(
    json.dumps(parser_config, indent=2),
    encoding="utf-8"
)


metadata = {
    "model_name": MODEL_NAME,
    "input_type": "binary_bag_of_words",
    "tokenizer_pattern": TOKEN_PATTERN,
    "input_size": len(vocabulary),
    "vocabulary": vocabulary,
    "classes": classes,
    "unknown_class": UNKNOWN_CLASS,
    "hidden_units": list(HIDDEN_UNITS),
    "dropout_rate": DROPOUT_RATE,
    "seed": SEED,
    "training_examples": len(X),
    "validation_examples": len(X_val),
    "held_out_test_examples": len(X_test),
    "validation_accuracy": float(val_accuracy),
    "test_accuracy": float(test_accuracy),
    "confidence_threshold_initial": MIN_CONFIDENCE,
    "initial_rejection_thresholds": {
        "minimum_confidence": MIN_CONFIDENCE,
        "minimum_margin": MIN_MARGIN,
        "minimum_token_coverage": MIN_TOKEN_COVERAGE
    },
    "end_to_end_held_out": dict(e2e_counts),
    "pipeline_test_count": len(pipeline_tests),
    "pipeline_tests_passed": pipeline_passed,
    "pipeline_test_accuracy": float(pipeline_accuracy),
    "motor_range_placeholder": {
        "minimum_degrees": MOTOR_MIN_DEGREES,
        "maximum_degrees": MOTOR_MAX_DEGREES,
        "warning": (
            "Verify the actual motor and mechanical range before "
            "using this value in hardware firmware."
        )
    },
    "quantization": quantization_status,
    "tflite_quantization_parameters": tflite_info,
    "tflite_test_accuracy": tflite_test_accuracy,
    "tflite_keras_agreement": tflite_keras_agreement,
    "tflite_pipeline_agreement": tflite_pipeline_agreement,
    "unknown_rejection_note": (
        "UNKNOWN is a trained class. It is NOT sufficient on its own: "
        "firmware must also apply the confidence, margin and vocabulary-"
        "coverage thresholds and the deterministic parameter validation "
        "described in command_parser_config.json."
    ),
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
# 14. WRITE FULL EVALUATION REPORT
# ============================================================

results_path = RESULTS_DIR / "intent_results.txt"

with results_path.open("w", encoding="utf-8") as file:
    file.write("AIEng4D INTENT MODEL RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Supported intents: {len(REQUIRED_CLASSES)}\n")
    file.write(f"Total classes: {len(classes)}\n")
    file.write(f"Training examples: {len(X)}\n")
    file.write(f"Validation examples: {len(X_val)}\n")
    file.write(f"Vocabulary size: {len(vocabulary)}\n")
    file.write(f"Held-out examples: {len(X_test)}\n")
    file.write(f"Network: {len(vocabulary)} -> {HIDDEN_UNITS[0]} -> "
               f"{HIDDEN_UNITS[1]} -> {len(classes)}\n")
    file.write(
        f"Validation accuracy: {val_accuracy * 100:.2f}%\n"
    )
    file.write(
        f"Classifier test accuracy: {test_accuracy * 100:.2f}%\n"
    )

    file.write("\nCLASSIFICATION REPORT (raw network, held-out)\n")
    file.write(report)

    file.write("\nCONFUSION MATRIX (rows=actual, columns=predicted)\n")
    file.write("Order: " + ", ".join(classes) + "\n")
    file.write(np.array2string(matrix))

    file.write("\n\nHELD-OUT MISCLASSIFICATIONS (raw network)\n")
    if misclassified_lines:
        file.write("\n".join(misclassified_lines) + "\n")
    else:
        file.write("None\n")

    file.write("\nEND-TO-END HELD-OUT (classifier + all rejection rules)\n")
    file.write(
        f"Supported commands accepted correctly: "
        f"{e2e_counts['ACCEPTED_OK']}/{e2e_supported_total}\n"
    )
    file.write(
        f"Supported commands rejected (safe, but annoying): "
        f"{e2e_counts['REJECTED_SUPPORTED']}/{e2e_supported_total}\n"
    )
    file.write(
        f"Supported commands accepted as WRONG intent (dangerous): "
        f"{e2e_counts['WRONG_INTENT_ACCEPTED']}/{e2e_supported_total}\n"
    )
    file.write(
        f"Unsupported text rejected correctly: "
        f"{e2e_counts['UNKNOWN_REJECTED']}/{e2e_unknown_total}\n"
    )
    file.write(
        f"Unsupported text FALSELY ACCEPTED (dangerous): "
        f"{e2e_counts['FALSE_ACCEPT']}/{e2e_unknown_total}\n"
    )

    file.write("\nEnd-to-end result per class:\n")
    for name in classes:
        item = e2e_per_class[name]
        file.write(f"  {name:13s} {item['ok']}/{item['total']}\n")

    file.write("\nEnd-to-end problems:\n")
    if e2e_problem_lines:
        file.write("\n".join(e2e_problem_lines) + "\n")
    else:
        file.write("None\n")

    file.write("\nPARSER COVERAGE (correct intent given, no classifier)\n")
    for (split_name, intent), (good, total) in parser_coverage.items():
        file.write(
            f"  {split_name:9s} {intent:13s} {good}/{total}\n"
        )

    file.write("\nPhrases the parser rejects even with the correct intent:\n")
    if parser_coverage_lines:
        file.write("\n".join(parser_coverage_lines) + "\n")
    else:
        file.write("None\n")

    file.write("\nCOMMAND PIPELINE TESTS\n")
    file.write("\n".join(pipeline_lines))

    file.write("\n\nPIPELINE TEST SUMMARY\n")
    file.write(
        f"Passed: {pipeline_passed}/{len(pipeline_tests)}\n"
    )
    file.write(
        f"Pipeline test score: {pipeline_accuracy:.2f}%\n"
    )

    file.write("\nTFLITE INT8 PARITY (held-out)\n")
    if tflite_test_accuracy is None:
        file.write("Not available (conversion failed)\n")
    else:
        file.write(
            f"INT8 classifier accuracy: "
            f"{tflite_test_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"Agreement with Keras predictions: "
            f"{tflite_keras_agreement * 100:.2f}%\n"
        )
        file.write(
            f"Same accept/reject decision as Keras: "
            f"{tflite_pipeline_agreement * 100:.2f}%\n"
        )
        file.write(f"Quantization parameters: {tflite_info}\n")

    file.write("\nMODEL FILE SIZES\n")
    file.write(
        f"Keras model: {model_path.stat().st_size:,} bytes\n"
    )

    if tflite_path.exists():
        file.write(
            f"INT8 TFLite: {tflite_path.stat().st_size:,} bytes\n"
        )

    file.write("\nQUANTIZATION\n")
    file.write(quantization_status + "\n")

    file.write("\nINDIVIDUAL PREDICTIONS (raw network, held-out)\n")
    file.write("\n".join(prediction_lines) + "\n")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Classifier test accuracy: {test_accuracy * 100:.2f}%")
print(
    f"End-to-end held-out: "
    f"{e2e_counts['ACCEPTED_OK']}/{e2e_supported_total} supported accepted, "
    f"{e2e_counts['WRONG_INTENT_ACCEPTED']} wrong-intent accepted, "
    f"{e2e_counts['UNKNOWN_REJECTED']}/{e2e_unknown_total} unsupported "
    f"rejected, {e2e_counts['FALSE_ACCEPT']} falsely accepted"
)
print(
    f"Command pipeline tests: "
    f"{pipeline_passed}/{len(pipeline_tests)} "
    f"({pipeline_accuracy:.2f}%)"
)

if tflite_test_accuracy is not None:
    print(
        f"INT8 TFLite: {tflite_test_accuracy * 100:.2f}% accuracy, "
        f"{tflite_keras_agreement * 100:.2f}% agreement with Keras"
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

if pipeline_failures:
    print("\nFAILED SAFETY TESTS:")
    for phrase in pipeline_failures:
        print("  -", repr(phrase))

    sys.exit(1)