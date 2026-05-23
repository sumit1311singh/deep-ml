import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    X, y, w0 = np.array(X), np.array(y), np.array(w0)
    result = []
    n=len(y)

    for i in range(n_steps):
        if n_steps==1:
            lr_i=10**a
        else:
            lr_i=10**(a+((b-a)*i)/(n_steps-1))
        y_hat = np.dot(X, w0)
        gradient = (2/n)*np.dot((X.T), (y_hat - y))
        w0 = w0 - lr_i*gradient
        y_hat = np.dot(X, w0)
        mse = np.mean((y_hat-y)**2)
        result.append(float(mse))
        #print(lr_i)
    #print(result)
    return result
