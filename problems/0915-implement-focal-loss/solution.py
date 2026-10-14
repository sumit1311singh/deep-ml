import torch

def focal_loss(logits, targets, gamma=2.0):
    # TODO: mean focal loss over the batch
    logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp = torch.exp(logits)

    prob = exp / torch.sum(exp, dim=1, keepdim=True)
    prob_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

    return -((1-prob_t)**gamma * torch.log(prob_t)).mean()
