import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, GlobalMaxPooling1D, Dropout, BatchNormalization
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping 



# DNA one‑hot encoding dictionary
DNA_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}


def encode_char(c):
    return DNA_MAP.get(c, [0,0,0,0])

MAX_LEN = 81   # choose your fixed length

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
            seq, label = line.strip().split(",")


            sequences.append(seq)
            labels.append(int(label))

    return sequences, labels


def encode_dataset(sequences):
    """Encode all sequences."""
    
    return np.array([encode_sequence(seq) for seq in sequences])


# ---------- MAIN PROGRAM ----------

training_sequences, training_labels = load_dataset("Data/training.txt")
testing_sequences, testing_labels = load_dataset("Data/testing.txt")

# augment training only

X_train = encode_dataset(training_sequences)
y_train = np.array(training_labels)
X_test = encode_dataset(testing_sequences)
y_test = np.array(testing_labels)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)



model = Sequential()

# Block 1
model.add(Conv1D(filters=32, kernel_size=24, activation="relu", padding="same", input_shape=(MAX_LEN, 4)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Block 2
model.add(Conv1D(filters=64, kernel_size=15, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Block 3
model.add(Conv1D(filters=128, kernel_size=10, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Block 4
model.add(Conv1D(256, kernel_size=6, activation="relu", padding="same"))
model.add(GlobalMaxPooling1D())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.125))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.004), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=80, callbacks=[early_stop])

model.save("promoter_cnn.keras")