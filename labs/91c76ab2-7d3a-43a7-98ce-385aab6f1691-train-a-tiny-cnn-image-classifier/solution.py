import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        # TODO: define conv, activation, pooling, and classifier layers
        # Input shape: (N, 1, img_size, img_size)
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64, n_classes)  
        )

    def forward(self, x):
        # TODO: implement forward pass; return logits of shape (N, n_classes)
        return self.model(x)


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    # TODO: return TinyCNN(img_size=img_size, n_classes=n_classes)
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model.

    Args:
        model: nn.Module from build_model
        train_x: FloatTensor (N, 1, H, W)
        train_y: LongTensor (N,)
        epochs: number of full passes over the data
        lr: optimizer learning rate
        batch_size: mini-batch size
        seed: RNG seed for shuffling / init determinism

    Returns:
        Trained model (same instance is fine).
    """
    # TODO:
    # - torch.manual_seed(seed)
    # - CrossEntropyLoss + Adam (or SGD)
    # - mini-batch loop for `epochs` epochs
    # - model.train(); zero_grad -> forward -> loss -> backward -> step
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay=0
    ) # fix params

    model.train()
    # have batches as well
    for epoch in range(epochs):
        idx = torch.randperm(len(train_x))

        shuffled_train_x, shuffled_train_y = train_x[idx], train_y[idx] 

        for start in range(0, len(train_x), batch_size):
            optimizer.zero_grad()

            x_batch = shuffled_train_x[start:start+batch_size]
            y_batch = shuffled_train_y[start:start+batch_size]


            logits = model(x_batch) # fix this

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

    with torch.no_grad():
        model.eval()

    return model
