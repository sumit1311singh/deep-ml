import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # squared L2 distances
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    
    # triplet loss per sample
    losses = F.relu(pos_dist - neg_dist + margin)
    
    # mean over batch
    return losses.mean()
