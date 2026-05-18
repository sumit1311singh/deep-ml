def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    if num_steps==0:
        return []
    elif num_steps==1:
        return [initial_lr]
    decay_rate = (end_lr-initial_lr)/(num_steps-1)
    #print(decay_rate)
    lr_list = []
    for i in range(num_steps):
        lr_list.append(initial_lr)
        initial_lr+=decay_rate
    return lr_list