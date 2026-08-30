
import numpy as np

def matrix_image(A):
    M = A.astype(float).copy()
    m, n = M.shape
    pivot_row = 0
    pivot_cols = []
    for j in range(n):
        # Step 3A: find a row ≥ pivot_row with a nonzero entry in column j
        pivot_candidate = None
        for i in range(pivot_row, m):
            if M[i, j] != 0:
                pivot_candidate = i
                break
		# Step 3B: if no pivot found in this column, skip it
        if pivot_candidate is None:
            continue

        # Step 4A: swap pivot_candidate row with pivot_row
        if pivot_candidate != pivot_row:
            M[[pivot_row, pivot_candidate]] = M[[pivot_candidate, pivot_row]]

        # Step 4B: record this column as a pivot column
        pivot_cols.append(j)

		# Step 4C: eliminate entries below the pivot
        pivot_val = M[pivot_row, j]
        for i in range(pivot_row + 1, m):
            if M[i, j] != 0:
                factor = M[i, j] / pivot_val
                M[i] = M[i] - factor * M[pivot_row]
                
        pivot_row += 1
        if pivot_row == m:
            break 

	# Step 5: extract pivot columns from the original matrix A
    return A[:, pivot_cols]