import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    X=np.array(X)

    X_cen=X-np.mean(X, axis=0)

    cov_mat = (X_cen.T @ X_cen)/(X_cen.shape[0]-1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    eigenvalues_sum = np.sum(eigenvalues)
    result = [eigenvalue/eigenvalues_sum for eigenvalue in eigenvalues]

    result = np.sort(result)[::-1]

    return result