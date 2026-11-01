import numpy as np

def unet_time_embedding(timesteps: list, embed_dim: int, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, max_period: int = 10000) -> np.ndarray:
    """
    Compute time embeddings for a diffusion model U-Net.
    
    Args:
        timesteps: list or 1D array of shape (B,) with timestep values
        embed_dim: dimension of sinusoidal embedding (must be even)
        W1: weight matrix of first linear layer, shape (embed_dim, hidden_dim)
        b1: bias of first linear layer, shape (hidden_dim,)
        W2: weight matrix of second linear layer, shape (hidden_dim, output_dim)
        b2: bias of second linear layer, shape (output_dim,)
        max_period: controls the frequency range for sinusoidal embedding
    
    Returns:
        numpy array of shape (B, output_dim) with time embeddings
    """
    timesteps = np.asarray(timesteps, dtype=np.float64)

    half_dim = embed_dim // 2

    frequencies = np.exp(
        -np.log(max_period) *
        np.arange(half_dim) / half_dim
    )

    args = timesteps[:, None] * frequencies[None, :]
    embedding = np.concatenate(
        [np.sin(args), np.cos(args)],
        axis=-1
    )

    hidden = embedding @ W1 + b1
    hidden = hidden / (1.0 + np.exp(-hidden))

    output = hidden @ W2 + b2

    return output