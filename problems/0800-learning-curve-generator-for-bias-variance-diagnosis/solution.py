import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train, y_train, X_val, y_val, train_sizes = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(train_sizes)

    b_val=np.array(y_val)
    A_val=np.ones((len(b_val), degree+1))
    train_sizes = sorted(train_sizes)

    for i in range(degree + 1):
        A_val[:, i] = np.array(X_val.T)**i 

    train_errors = []
    val_errors = []
    diagnosis = 'good_fit'

    for n in train_sizes:
        b=np.array(y_train[:n])
        A=np.ones((len(b), degree+1))

        for i in range(degree + 1):
            A[:, i] = np.array(X_train[:n].T)**i

        theta = np.linalg.pinv(A) @ b

        y_train_pred = A @ theta
        train_error = np.mean((b - y_train_pred)**2)
        # Clip floating‑point noise
        if abs(train_error) < 1e-12:
            train_error = 0.0
        train_errors.append(train_error)

        y_val_pred = A_val @ theta
        val_error = np.mean((y_val - y_val_pred)**2)
        if abs(val_error) < 1e-12:
            val_error = 0.0
        val_errors.append(val_error)

    if train_errors[-1] > bias_threshold:
        diagnosis = 'high_bias'
    elif val_errors[-1] - train_errors[-1] > variance_threshold:
        diagnosis = 'high_variance'

    return {'train_errors': train_errors, 'val_errors': val_errors, 'diagnosis': diagnosis}

