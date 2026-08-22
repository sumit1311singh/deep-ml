import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    A = A.astype(float).copy()
    m, n = A.shape
    rank = 0

    for col in range(n):
        # Find pivot row (largest absolute value in this column)
        pivot_row = np.argmax(np.abs(A[rank:, col])) + rank

        # If pivot is below tolerance → skip column
        if abs(A[pivot_row, col]) < tol:
            continue

        # Swap pivot row into position
        A[[rank, pivot_row]] = A[[pivot_row, rank]]

        # Eliminate rows below pivot
        for r in range(rank + 1, m):
            factor = A[r, col] / A[rank, col]
            A[r, col:] -= factor * A[rank, col:]

        rank += 1

        if rank==m:
            break

    return rank