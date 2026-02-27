def nucleotides_to_vectors(nucleotide_string):
    """
    Convert a string of nucleotides to 4-digit one-hot encoded vectors.
    
    Args:
        nucleotide_string (str): String containing nucleotides (A, T, G, C)
    
    Returns:
        list: List of one-hot encoded vectors (each as a list of 4 digits)
    """
    nucleotide_map = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'C': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # Optional: Handle 'N' for unknown nucleotides
    }
    
    vectors = []
    for nucleotide in nucleotide_string.upper():
        if nucleotide in nucleotide_map:
            vectors.append(nucleotide_map[nucleotide])
        else:
            raise ValueError(f"Invalid nucleotide: {nucleotide}")
    
    return vectors


