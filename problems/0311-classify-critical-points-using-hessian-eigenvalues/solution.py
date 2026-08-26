import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    H = np.array(hessian, float)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(H)

    # Sort them (Deep‑ML expects sorted eigenvalues)
    eigenvalues = np.sort(eigenvalues)

    # Check for near‑zero eigenvalues → inconclusive
    if np.any(np.abs(eigenvalues) < tol):
        return None

    # All positive → local minimum
    if np.all(eigenvalues > 0):
        return -1

    # All negative → local maximum
    if np.all(eigenvalues < 0):
        return +1

    # Mixed signs → saddle point
    return 0
