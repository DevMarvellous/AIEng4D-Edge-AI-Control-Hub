    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================
# HELD-OUT EVALUATION
# ============================================================

test_texts = []
test_labels = []
for intent in classes:
    for phrase in test_data[intent]:
        test_texts.append(phrase)
        test_labels.append(class_to_id[intent])

X_test = np.asarray([encode_text(text) for text in test_texts], dtype=np.float32)
y_test = np.asarray(test_labels, dtype=np.int32)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
test_scores = model.predict(X_test, verbose=0)
test_predictions = np.argmax(test_scores, axis=1)

report = classification_report(
    y_test,
    test_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
)
matrix = confusion_matrix(y_test, test_predictions)

print("\n==============================")
print("HELD-OUT TEST RESULTS")
print("==============================")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification report:")
print(report)
print("Confusion matrix:")
print(matrix)
print("\nIndividual predictions:")

prediction_lines = []
for phrase, actual, predicted, scores in zip(
    test_texts, y_test, test_predictions, test_scores
):
    confidence = float(scores[predicted])
    line = (
        f"{phrase:62s} | actual={classes[actual]:12s} | "
        f"predicted={classes[predicted]:12s} | confidence={confidence:.2%}"
    )
    print(line)
    prediction_lines.append(line)

# ============================================================
# SAVE DEPLOYMENT FILES
# ============================================================

model_path = os.path.join(OUTPUT_DIR, "intent_model.keras")
model.save(model_path)

with open(os.path.join(OUTPUT_DIR, "vocabulary.json"), "w", encoding="utf-8") as f:
    json.dump(vocabulary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
    json.dump(classes, f, indent=2)

metadata = {
    "model_name": "AIEng4D intent classifier",
    "input_type": "binary_bag_of_words",
    "tokenizer_pattern": "[a-z0-9]+",
    "input_size": len(vocabulary),
    "vocabulary": vocabulary,
    "classes": classes,
    "seed": SEED,
    "training_examples": len(X),
    "held_out_test_examples": len(X_test),
    "test_accuracy": float(test_accuracy),
    "confidence_threshold_initial": 0.70,
    "unknown_rejection_note": (
        "NO_ACTION is not a class. Unknown-command rejection must be "
        "implemented separately using command validation and a confidence "
        "threshold. Softmax confidence alone does not guarantee unknown detection."
    ),
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

results_path = os.path.join(RESULTS_DIR, "intent_training_results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("AIEng4D INTENT MODEL RESULTS\n")
    f.write("============================\n")
    f.write(f"Intent classes: {len(classes)}\n")
    f.write(f"Training phrases: {len(X)}\n")
    f.write(f"Vocabulary size: {len(vocabulary)}\n")
    f.write(f"Held-out test phrases: {len(X_test)}\n")
    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write(report)
    f.write("\nCONFUSION MATRIX\n")
    f.write(np.array2string(matrix))
    f.write("\n\nINDIVIDUAL PREDICTIONS\n")
    f.write("\n".join(prediction_lines))

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print("Saved model:", model_path)
print("Saved vocabulary:", os.path.join(OUTPUT_DIR, "vocabulary.json"))
print("Saved classes:", os.path.join(OUTPUT_DIR, "classes.json"))
print("Saved metadata:", os.path.join(OUTPUT_DIR, "metadata.json"))
print("Saved evaluation:", results_path)
print("\nNOTE: Unknown-command rejection is not learned as a tenth class.")
print("It will be implemented separately in the ESP32 command-validation layer.")
