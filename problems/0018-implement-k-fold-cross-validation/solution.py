import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    samples = np.arange(n_samples)
    result = []

    if shuffle:
        np.random.shuffle(samples)

    fold_size = n_samples//k
    extra = n_samples%k

    for i in range(k):
        start = i*fold_size
        end = start+fold_size + extra
        
        train = np.concatenate((
            samples[:start], 
            samples[end:])
        )
        test = samples[start:end]

        result.append((train, test))

    return result
