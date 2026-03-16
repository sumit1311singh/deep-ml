def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    if x==0.0 or n==0.0:
        return 0.0
    return c*n*(x**(n-1))