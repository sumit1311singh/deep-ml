import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions, y_true = np.array(predictions), np.array(y_true)
    
    mean_pred = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_pred - y_true)**2)

    variance = np.mean(np.mean((predictions - mean_pred)**2, axis=0))

    mse = bias_squared + variance

    return {'bias_squared': bias_squared, 'variance': variance, 'mse': mse}

