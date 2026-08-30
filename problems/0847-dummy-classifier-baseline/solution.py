import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """

    y_train = np.array(y_train)
    preds = []

    # Unique classes sorted
    classes, counts = np.unique(y_train, return_counts=True)
    sorted_classes = classes.tolist()

    # Strategy: most_frequent
    if strategy == "most_frequent":
        max_count = counts.max()
        # classes with max_count
        candidates = classes[counts == max_count]
        # break ties by smallest label
        most_freq = int(candidates.min())
        return [most_freq] * n_test

    # Strategy: constant
    if strategy == "constant":
        return [constant] * n_test

    # Strategy: uniform
    if strategy == "uniform":
        k = len(sorted_classes)
        return [sorted_classes[i % k] for i in range(n_test)]

    # Strategy: stratified
    if strategy == "stratified":
        total = len(y_train)
        freqs = counts / total

        # floor allocations
        floors = [int(np.floor(n_test * f)) for f in freqs]

        # fractional parts
        fracs = [(n_test * f) - np.floor(n_test * f) for f in freqs]

        allocated = sum(floors)
        remaining = n_test - allocated

        # distribute remaining to largest fractional parts
        # ties broken by smallest class label → sorted_classes order already sorted
        order = np.argsort(-np.array(fracs))  # descending fractional part

        extra = [0] * len(classes)
        for idx in order[:remaining]:
            extra[idx] += 1

        final_counts = [floors[i] + extra[i] for i in range(len(classes))]

        # output grouped in sorted class order
        preds = []
        for cls, cnt in zip(sorted_classes, final_counts):
            preds.extend([cls] * cnt)

        return preds

    # Unknown strategy
    raise ValueError("Unknown strategy")
