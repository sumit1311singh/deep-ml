from typing import Tuple
import numpy as np

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    val_losses = np.array(val_losses)
    best_loss = val_losses[0] 
    best_idx = 0
    curr_patience = patience
    stopping = 0
    for i in range(len(val_losses)):
        improvement = best_loss - val_losses[i]
        #print(i, improvement, best_loss, "fefe")
        if improvement>min_delta:
            best_loss = val_losses[i]
            best_idx = i 
            curr_patience = patience
        else:
            if i!=0:
                curr_patience -= 1
                #print(i, curr_patience, "ededed")
                if curr_patience==0:
                    stopping = i
                    #print(i, "break")
                    break
    if best_idx == len(val_losses)-1:
        stopping=best_idx
    return (stopping, best_idx)
