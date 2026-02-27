import numpy as np

sequences = []
labels = []

with open("training.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        label_symbol, sequence = line.split("\t")
        
        # Convert label to 0/1
        if label_symbol == "-":
            label = 0
        else:
            label = 1
        
        sequences.append(sequence)
        labels.append(label)

print(sequences)  # first 50 bases
print(labels)  # corresponding labels