import numpy as np

def lu_decomposition(A: list) -> tuple:
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Compute U[i, j]
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        # Compute L[j, i]
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U