import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("promoter_cnn.h5")

DNA_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}

MAX_LEN = 70

def encode_char(c):
    return DNA_MAP.get(c, [0,0,0,0])

def encode_sequence(seq):
    encoded = [encode_char(c) for c in seq]

    if len(encoded) < MAX_LEN:
        encoded += [[0,0,0,0]] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return np.array(encoded)

def predict_sequence(seq):
    encoded = encode_sequence(seq)
    encoded = np.expand_dims(encoded, axis=0)  # shape: (1,100,4)

    prediction = model.predict(encoded)[0][0]

    print("Promoter probability:", prediction)

    if prediction > 0.5:
        print("✅ Promoter detected")
        return True
    else:
        print("❌ Not a promoter")
        return False

def find_promoter_location(seq, window_size=100):
    best_score = 0
    best_position = -1

    for i in range(len(seq) - window_size + 1):
        chunk = seq[i:i+window_size]

        encoded = encode_sequence(chunk)
        encoded = np.expand_dims(encoded, axis=0)

        score = model.predict(encoded)[0][0]

        if score > best_score:
            best_score = score
            best_position = i

    return best_position, best_score


if __name__ == "__main__":
    sequence = input("Enter DNA sequence: ").upper()

    print("\n--- Classification ---")
    if (predict_sequence(sequence)):
        print("\n--- Promoter Location ---")
        position, score = find_promoter_location(sequence)
        print(f"Best promoter location: {position} (score: {score:.4f})")
    else:
        print("No promoter found, skipping location search.")
