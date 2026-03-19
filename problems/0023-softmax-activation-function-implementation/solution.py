import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    arr = np.array(scores)
    result = []
    denominator = np.sum(np.exp(arr))
    for i in arr:
        result.append(np.exp(i)/denominator)
    return np.round(result, 4)