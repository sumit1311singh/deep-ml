import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    optimizer.zero_grad()
    K = len(micro_batches)
    total_loss = 0.0

    for micro_batch in micro_batches:
        x, y = micro_batch

        output = model(x)
        loss = criterion(output, y)

        total_loss+=loss.item()

        (loss/K).backward()

    optimizer.step()

    return total_loss/K
