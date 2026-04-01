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

MAX_LEN = 100

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
    else:
        print("❌ Not a promoter")


if __name__ == "__main__":
    sequence = input("Enter DNA sequence: ").upper()

    print("\n--- Classification ---")
    predict_sequence(sequence)
