import numpy as np

def check_positive_definite(matrix: list) -> dict:
    M = np.array(matrix, float)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(M)

    # Sort ascending
    eigenvalues_sorted = np.sort(eigenvalues)

    # Round to 4 decimals (Deep‑ML requirement)
    eigenvalues_rounded = np.round(eigenvalues_sorted, 4)

    # Positive definite ⇢ all eigenvalues strictly > 0
    is_pd = np.all(eigenvalues_rounded > 0)

    return {
        'is_positive_definite': bool(is_pd),
        'eigenvalues': eigenvalues_rounded.tolist()
    }
