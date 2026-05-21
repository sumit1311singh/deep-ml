import numpy as np

def smooth_labels(y_true, num_classes, epsilon):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """
    y_true=np.array(y_true)
    N=len(y_true)

    smoothed_labels = np.full(
        (N, num_classes),
        epsilon/num_classes
    )

    smoothed_labels[np.arange(N), y_true] = (
        1-epsilon+(epsilon/num_classes)
    )

    return smoothed_labels



def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    logits, y_true = np.array(logits), np.array(y_true)
    N = logits.shape[0]
    
    smoothed_labels = smooth_labels(y_true, num_classes, epsilon)
    
    logits_stable = logits - np.max(
        smoothed_labels, 
        axis=1,
        keepdims=True
    ) 

    exp_logits = np.exp(logits_stable)

    exp_sum = np.sum(
        exp_logits,
        axis=1,
        keepdims=True
    )

    log_probs = (
        logits_stable - np.log(exp_sum)
    )

    losses = -np.sum(
        smoothed_labels * log_probs,
        axis=1
    )

    loss = np.mean(losses)

    if round_decimals is not None:
        loss = np.round(loss, round_decimals)

    return float(loss)