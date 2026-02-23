import re
import sys
from pdfminer.high_level import extract_text


def extract_sequences_from_pdf(input_pdf, output_txt):
    text = extract_text(input_pdf)

    lines = text.splitlines()

    current_sign = None
    current_sequence = []

    with open(output_txt, "w") as out:
        for line in lines:
            line = line.strip()

            # Header line
            if line.startswith(">"):
                # Save previous sequence
                if current_sign and current_sequence:
                    seq = "".join(current_sequence)
                    seq = re.sub(r'^N+', '', seq)  # remove leading Ns
                    out.write(f"{current_sign}\t{seq}\n")

                # Extract sign inside parentheses
                match = re.search(r'\((.*?)\)', line)
                current_sign = match.group(1) if match else None
                current_sequence = []

            # DNA line (only ACGTN letters)
            elif re.fullmatch(r'[ACGTN]+', line):
                current_sequence.append(line)

        # Save last sequence
        if current_sign and current_sequence:
            seq = "".join(current_sequence)
            seq = re.sub(r'^N+', '', seq)
            out.write(f"{current_sign}\t{seq}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.pdf output.txt")
    else:
        extract_sequences_from_pdf(sys.argv[1], sys.argv[2])