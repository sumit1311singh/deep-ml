import numpy as np

def dice_score(y_true, y_pred):
    # Count ones in each array
    a = np.sum(y_true)
    b = np.sum(y_pred)

    # If either array has no ones → Dice = 0.0
    if a == 0 or b == 0:
        return 0.0

    # True positives = intersection of ones
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # Dice formula from the page:
    # Dice = 2 * TP / (2*TP + FP + FN)
    res = (2 * tp) / (a + b)

    return round(res, 3)
