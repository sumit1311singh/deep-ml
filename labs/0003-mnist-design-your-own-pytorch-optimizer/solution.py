import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    Design your own optimizer!
    - You can base it on SGD, RMSProp, Adam, or create something new.
    - Must subclass torch.optim.Optimizer.
    - Only dense gradients are supported.
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        # You can add your own hyperparameters here
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                # --- TODO: implement your own update rule ---
                # Example: simple SGD update
                
                state = self.state[p]

                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(grad)
                
                state['square_avg'].mul_(group['alpha']).add_(
                    grad.square(), alpha=1 - group['alpha']
                )

                normalized_grad = grad / (state['square_avg'].sqrt() + group['eps'])

                p.add_(normalized_grad, alpha=-lr)

        return loss
