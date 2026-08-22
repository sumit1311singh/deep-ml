import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination
    for i in range(n):
        # Partial pivoting: pick row with largest pivot
        pivot = i + np.argmax(np.abs(A[i:, i]))
        if A[pivot, i] == 0:
            raise ValueError("Matrix is singular.")

        # Swap rows in A and b
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        # Eliminate rows below pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x
