from collections import Counter
import numpy as np


def sequence_entropy(seq):
    """Shannon entropy of a protein sequence, as implemented in ProFET (Ofer & Linial, 2015)"""
    length = len(seq)
    freq = Counter(seq)
    entropy = 0.0

    for word in freq:
        probability = freq[word] / (1.0 * length)
        self_information = np.log2(1.0 / probability)
        entropy += (probability * self_information)

    return entropy
