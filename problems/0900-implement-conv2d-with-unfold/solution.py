import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # TODO: implement conv2d using F.unfold + matmul
    N, C_in, H, width = x.shape
    C_out, _, kh, kw = W.shape

    patches = F.unfold(
        x, 
        kernel_size=(kh, kw),
        stride=stride, 
        padding=padding
    )

    w_reshaped = torch.reshape(W, (C_out, C_in*kh*kw))

    matmul_result = torch.matmul(w_reshaped, patches)

    if b is not None:
        matmul_result = matmul_result + b.reshape(1, C_out, 1)

    H_out = (H + 2*padding - kh)//stride + 1
    W_out = (width + 2*padding - kw)//stride + 1

    matmul_result_reshaped = torch.reshape(matmul_result, (N, C_out, H_out, W_out))

    return matmul_result_reshaped
