import numpy as np

def rref(A):
    M = A.astype(float).copy()
    m, n = M.shape

    pivot_row = 0
    pivot_col = 0

    while pivot_row < m and pivot_col < n:

        # Step 1: Find pivot in column pivot_col
        pivot = None
        for r in range(pivot_row, m):
            if M[r, pivot_col] != 0:
                pivot = r
                break

        # If no pivot in this column, move to next column
        if pivot is None:
            pivot_col += 1
            continue

        # Step 2: Swap pivot row into position
        if pivot != pivot_row:
            M[[pivot_row, pivot]] = M[[pivot, pivot_row]]

        # Step 3: Normalize pivot row (make pivot = 1)
        pivot_val = M[pivot_row, pivot_col]
        M[pivot_row] = M[pivot_row] / pivot_val

        # Step 4: Eliminate all other rows in pivot column
        for r in range(m):
            if r != pivot_row and M[r, pivot_col] != 0:
                factor = M[r, pivot_col]
                M[r] = M[r] - factor * M[pivot_row]

        # Move to next pivot row and column
        pivot_row += 1
        pivot_col += 1

    return M
