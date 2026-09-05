import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int):
    # Add bias column
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])

    n, d = Xb.shape
    theta = np.zeros(d)

    losses = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(iterations):
        # Forward pass
        z = Xb @ theta
        p = sigmoid(z)

        # Binary cross entropy **SUM** (Deep‑ML requirement)
        loss = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        losses.append(round(loss, 4))

        # Gradient (same as mean gradient because sum loss is used)
        grad = Xb.T @ (p - y)

        # Update
        theta -= learning_rate * grad

    # Round theta to 4 decimals
    theta = [round(t, 4) for t in theta]

    return theta, losses
