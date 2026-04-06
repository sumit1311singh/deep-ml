import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Define functions and their derivatives
    func_map = {
        'square': (lambda u: u**2, lambda u: 2*u),
        'sin':    (np.sin, lambda u: np.cos(u)),
        'exp':    (np.exp, lambda u: np.exp(u)),
        'log':    (np.log, lambda u: 1/u)
    }
    
    # Start with input value
    val = x
    grad = 1.0
    
    # Apply functions right-to-left
    for fn in reversed(functions):
        f, fprime = func_map[fn]
        grad *= fprime(val)   # chain rule multiplier
        val = f(val)          # update value for next outer function
    
    return grad