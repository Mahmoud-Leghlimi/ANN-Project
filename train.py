import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout


# DNA one‑hot encoding dictionary
DNA_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}
def encode_char(c):
    return DNA_MAP.get(c, [0,0,0,0])

MAX_LEN = 100   # choose your fixed length

def encode_sequence(seq):
    encoded = [encode_char(c) for c in seq]

    if len(encoded) < MAX_LEN:
        encoded += [[0,0,0,0]] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return encoded


def load_dataset(filepath: str):
    """Load dataset and extract labels + sequences."""
    
    sequences = []
    labels = []

    with open(filepath, "r") as file:
        for line in file:
            label, seq = line.strip().split("\t")

            sequences.append(seq)
            labels.append(1 if label == "+" else 0)

    return sequences, labels


def encode_dataset(sequences):
    """Encode all sequences."""
    
    return np.array([encode_sequence(seq) for seq in sequences])


# ---------- MAIN PROGRAM ----------

training_sequences, training_labels = load_dataset("Data/training.txt")
testing_sequences, testing_labels = load_dataset("Data/testing.txt")

X_train = encode_dataset(training_sequences)
y_train = np.array(training_labels)
X_test = encode_dataset(testing_sequences)
y_test = np.array(testing_labels)

# print("Training dataset shape:", X_train.shape)
# print("Training labels shape:", y_train.shape)
# print("Testing dataset shape:", X_test.shape)
# print("Testing labels shape:", y_test.shape)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=12, activation="relu", input_shape=(MAX_LEN, 4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=6, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save("promoter_cnn.h5")