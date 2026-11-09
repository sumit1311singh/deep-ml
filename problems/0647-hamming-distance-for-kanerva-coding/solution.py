import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
    """
    Compute Hamming distances and find active prototypes for Kanerva coding.

    Args:
        state: Binary state vector (list of 0s and 1s).
        prototypes: List of binary prototype vectors.
        threshold: Maximum Hamming distance for a prototype to be active.

    Returns:
        Tuple of (distances, active_indices).
    """
    distances = []
    active_indices = []

    for i, prototype in enumerate(prototypes):
        distance = sum(s != p for s, p in zip(state, prototype))
        distances.append(distance)

        if distance <= threshold:
            active_indices.append(i)

    return distances, active_indices