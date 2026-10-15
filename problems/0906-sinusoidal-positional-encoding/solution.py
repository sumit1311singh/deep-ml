import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.arange(d_model // 2, dtype=torch.float32)

    exp_div_term = torch.exp(-math.log(10000.0) * 2 * div_term / d_model)

    angles = position * exp_div_term

    encoding = torch.zeros(seq_len, d_model)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)

    return encoding


