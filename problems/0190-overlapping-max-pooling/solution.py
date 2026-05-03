import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    x = x.astype(float) 
    N, C, H, W = x.shape
    p, s = kernel_size, stride

    # ceil output size
    out_h = int(np.ceil((H - p) / s)) + 1
    out_w = int(np.ceil((W - p) / s)) + 1

    # required padding
    pad_h = max((out_h - 1) * s + p - H, 0)
    pad_w = max((out_w - 1) * s + p - W, 0)

    # pad with -inf for max pooling
    x = np.pad(x, ((0,0),(0,0),(0,pad_h),(0,pad_w)), constant_values=-np.inf)

    # sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(
        x, (p, p), axis=(2, 3)
    )

    # apply stride
    windows = windows[:, :, ::s, ::s, :, :]
    
    return windows.max(axis=(-2, -1)).astype(int)