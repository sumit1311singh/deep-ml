import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if training:
        mask = torch.rand(x.shape) > p
        return x*mask/(1-p)
    else:
        return x