import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x, y = np.array(x), np.array(y)

    b=np.array(y)
    A=np.ones((len(b), degree + 1))

    for i in range(degree + 1):
        A[:, i] = np.array(x)**i 
    
    lsq = np.linalg.lstsq(A, b.T, rcond=None)

    return lsq[0].tolist()

