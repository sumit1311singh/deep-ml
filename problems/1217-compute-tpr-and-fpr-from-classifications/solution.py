import numpy as np
from collections import Counter

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    # TODO: compute TP, FN, FP, TN then TPR and FPR
    counts = Counter(zip(y_true, y_pred))

    tp = counts.get((1, 1), 0)
    fn = counts.get((1, 0), 0)
    fp = counts.get((0, 1), 0)
    tn = counts.get((0, 0), 0)

    if tp+fn==0:
        tpr=0.0
    else:
        tpr = tp/(tp+fn)

    if fp+tn==0:
        fpr=0.0
    else:
        fpr = fp/(fp+tn)

    return tpr, fpr
