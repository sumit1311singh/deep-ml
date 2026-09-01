import numpy as np
from collections import Counter

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """

    y_true, y_scores = np.array(y_true), np.array(y_scores)

    mask = y_scores>=threshold
    y_scores[mask] = 1
    y_scores[~mask] = 0
    
    pairs = [tuple((i, j)) for i, j in zip(y_true, y_scores)]
    counts = Counter(pairs)

    #print(y_true)
    #print(y_scores)
    
    tp = counts.get((1, 1), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    
    precision = 0.0
    recall = 0.0
    
    if(tp+fp>0):
        precision = tp/(tp+fp)
    if(tp+fn>0):
        recall = tp/(tp+fn)
        
    return np.round([precision, recall], 4)


