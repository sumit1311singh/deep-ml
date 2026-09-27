import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    ds = TensorDataset(X, y)

    loader = DataLoader(ds, batch_size=4, shuffle=False)
    num_batches = len(loader)

    first_batch_X, first_batch_y = next(iter(loader))
    first_batch_X_shape = tuple(first_batch_X.shape)
    
    return num_batches, first_batch_X_shape