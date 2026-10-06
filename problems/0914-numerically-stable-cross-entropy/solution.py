import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    shifted_logits = logits - logits.max(dim=-1, keepdims=True).values

    exp_shifted = torch.exp(shifted_logits)

    log_sum_exp = torch.log(torch.sum(exp_shifted, dim=1))

    target_logits = shifted_logits.gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)

    log_probs = target_logits - log_sum_exp

    return -log_probs.mean()