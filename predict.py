import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

model = load_model("promoter_cnn.keras")

DNA_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}

MAX_LEN = 81

def encode_char(c):
    return DNA_MAP.get(c, [0, 0, 0, 0])

def encode_sequence(seq):
    encoded = [encode_char(c) for c in seq]

    if len(encoded) < MAX_LEN:
        encoded += [[0, 0, 0, 0]] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return np.array(encoded)

def predict_sequence(seq):
    encoded = encode_sequence(seq)
    encoded = np.expand_dims(encoded, axis=0)

    prediction = model.predict(encoded, verbose=0)[0][0]
    return prediction

def evaluate_file(filename):
    sequences = []
    labels = []

    with open(filename, "r") as f:
        for line in f:
            try:
                seq, label = line.strip().split(",")
                sequences.append(seq.upper())
                labels.append(int(label))
            except:
                continue

    # Encode all at once
    X = np.array([encode_sequence(seq) for seq in sequences])

    # Batch prediction (FAST)
    predictions = model.predict(X, verbose=0).flatten()

    correct = 0

    for i in range(len(sequences)):
        pred_label = 1 if predictions[i] > 0.5 else 0
        is_correct = pred_label == labels[i]

        print(
            f"Seq: {sequences[i][:10]}... | "
            f"Pred: {pred_label} ({predictions[i]:.3f}) | "
            f"True: {labels[i]} | "
            f"{'✔' if is_correct else '❌'}"
        )

        if is_correct:
            correct += 1

    accuracy = correct / len(sequences)

    print("\n--- Results ---")
    print(f"Total: {len(sequences)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_file("Data/testing.txt")