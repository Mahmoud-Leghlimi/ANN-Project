import numpy as np


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
    encoded = [encode_char(c) for c in seq]   # your encoding method
    
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))   # pad with zeros
    else:
        encoded = encoded[:MAX_LEN]                 # truncate
        
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

sequences, labels = load_dataset("training.txt")

X = encode_dataset(sequences)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print(X)