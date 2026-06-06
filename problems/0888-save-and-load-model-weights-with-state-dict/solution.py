import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # Create an in-memory buffer
    buffer = io.BytesIO()
    
    # Save src's state_dict into the buffer
    torch.save(src.state_dict(), buffer)
    
    # Rewind the buffer cursor to the beginning
    buffer.seek(0)
    
    # Load the state_dict back from buffer
    state_dict = torch.load(buffer)
    
    # Copy into dst
    dst.load_state_dict(state_dict)
    
    return dst
