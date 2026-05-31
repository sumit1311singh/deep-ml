import torch

def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    # convert to tensor for PyTorch ops
    step_t = torch.tensor(step, dtype=torch.float32)
    warmup_t = torch.tensor(warmup_steps, dtype=torch.float32)
    total_t = torch.tensor(total_steps, dtype=torch.float32)
    base_lr_t = torch.tensor(base_lr, dtype=torch.float32)

    if step <= warmup_steps:
        return (step_t / warmup_t * base_lr_t).item()

    progress = (step_t - warmup_t) / (total_t - warmup_t)
    progress = torch.clamp(progress, 0.0, 1.0)
    lr = 0.5 * (1 + torch.cos(torch.pi * progress)) * base_lr_t
    return lr.item()
