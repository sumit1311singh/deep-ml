import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    x = input_data.copy()

    for layer in range(num_layers):
        kernel = kernels[layer]
        kh, kw, in_channels, out_channels = kernel.shape

        if in_channels != x.shape[-1]:
            raise ValueError("Kernel input channels do not match current feature channels")

        if (kh, kw) != kernel_size:
            raise ValueError("Kernel dimensions do not match kernel_size")

        # Apply ReLU
        activated = np.maximum(x, 0)

        N, H, W, C = activated.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Zero-pad spatial dimensions
        padded = np.pad(
            activated,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant"
        )

        # Convolution: stride 1, no bias
        out = np.zeros(
            (N, H, W, growth_rate),
            dtype=np.result_type(x.dtype, kernel.dtype)
        )

        for i in range(kh):
            for j in range(kw):
                patch = padded[:, i:i + H, j:j + W, :]
                out += np.einsum("nhwc,co->nhwo", patch, kernel[i, j])

        # Concatenate new features with existing features
        x = np.concatenate((x, out), axis=-1)

    return x