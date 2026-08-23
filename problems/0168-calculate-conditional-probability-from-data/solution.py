import numpy as np

def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    data = np.array(data)

    denom = 0
    num = 0

    for i, j in data:
      if i==x:
        denom+=1
        if j==y:
          num+=1

    if denom==0:
      return 0.0

    return np.round(float(num/denom), 4)