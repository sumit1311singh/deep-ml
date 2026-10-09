import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool  = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
            self.gap   = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(8, 10)
            self.act   = nn.ReLU()
        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.pool(x)
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))
            x = self.gap(x)      # -> (N, C2, 1, 1)
            x = x.flatten(1)     # -> (N, C2)
            return self.fc(x)
    return TinyNet()
