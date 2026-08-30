import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    # Your code here
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    if S.size < Vt.shape[0]:
        # pad with True (zero singular values)
        pad = np.ones(Vt.shape[0] - S.size, dtype=bool)
        mask = np.concatenate([S < tol, pad])
    else:
        mask = S < tol
    col_vec = Vt[mask].T
    return col_vec