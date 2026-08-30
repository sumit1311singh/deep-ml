import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n=data.shape[0]

    shuffled_idx = np.random.default_rng(seed).permutation(n)
    shuffled_data=data[shuffled_idx]

    train_end=int(n*train_frac)
    validation_end = train_end + int(n * validation_frac)

    train = shuffled_data[:train_end]
    validation = shuffled_data[train_end:validation_end]
    test = shuffled_data[validation_end:]
    
    return [train, validation, test]