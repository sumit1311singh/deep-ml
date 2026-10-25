import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []

    for v in vectors:
        v = np.array(v, dtype=float)

        # Remove projections onto existing basis vectors
        for u in basis:
            v = v - np.dot(v, u) * u

        # Check if anything remains (linear independence test)
        norm = np.linalg.norm(v)
        if norm > tol:
            basis.append(v / norm)

    return basis
