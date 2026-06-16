import numpy as np

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected (list or array-like)
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    # Convert to numpy arrays for convenience
    v = np.array(v, dtype=float)
    L = np.array(L, dtype=float)

    # Projection formula
    proj = (np.dot(v, L) / np.dot(L, L)) * L
    return proj.tolist()
