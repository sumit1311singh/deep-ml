import numpy as np

def differentiate(poly: list[float]) -> list[float]:
    n=len(poly)-1
    derivative = [float((n-i)*poly[i]) for i in range(n)]
    return derivative if derivative else [0.0]

def evaluate(poly: list[float], x: float) -> float:
    return float(np.polyval(poly, x))

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g_prime = differentiate(g_coeffs)
    h_prime = differentiate(h_coeffs)

    g_x = evaluate(g_coeffs, x)
    h_x = evaluate(h_coeffs, x)
    g_prime_x = evaluate(g_prime, x)
    h_prime_x = evaluate(h_prime, x)

    #print(g_x, h_x, g_prime_x, h_prime_x)

    numerator = g_prime_x * h_x - g_x * h_prime_x
    denominator = h_x**2
    #print(numerator, denominator)
    return numerator/denominator