import torch
import torch.nn as nn


def apply_conv2d():
    """Build nn.Conv2d(1, 1, kernel_size=2, bias=False), set a fixed kernel, convolve a fixed input.

    Under torch.no_grad(), set weight to tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).
    Input is tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]).

    Returns:
        torch.Tensor: Output of shape (1, 1, 2, 2).
    """
    # TODO: build conv, set weight under no_grad, apply to fixed input
    input_tensor = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    kernel = torch.tensor([[[[1.0, 0.0],[0.0, 1.0]]]])
    with torch.no_grad():
        out = nn.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
        out.weight.copy_(kernel.reshape(1, 1, 2, 2))
    
        result = out(input_tensor)

    return result

