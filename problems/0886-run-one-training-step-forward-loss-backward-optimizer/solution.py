import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 1. Clear the grad accumulator
    optimizer.zero_grad()

    # 2. Forward Pass
    y_pred = model(x)

    # 3. Compute Loss
    loss = F.mse_loss(y_pred, y)

    # 4. Backward Pass
    loss.backward()

    # 5: Update params
    optimizer.step()
    
    return loss.item()