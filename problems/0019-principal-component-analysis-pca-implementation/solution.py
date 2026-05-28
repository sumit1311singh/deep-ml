import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Step 1: Standerdize the data
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    std_data = (data-data_mean)/data_std

    # Step 2: Find Cov Mat for the data
    cov_mat = (std_data.T @ std_data) / (std_data.shape[0] - 1)

    # Step 3: Find Eigen Values and Eigen Vectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step 4: Fix Eigen Vectors direction
    for i in range(eigen_vectors.shape[1]):
        vec = eigen_vectors[:, i]
        for val in vec:
            if (abs(val) > 1e-10):
                if val<0:
                    vec *= -1
                break


    # Step 5: Select PCs
    sorted_idx = np.argsort(eigen_values)
    sorted_idx = sorted_idx[::-1]
    eigen_values = eigen_values[sorted_idx]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    
    eigen_vectors = eigen_vectors[:, :k]
    pcs = np.round(eigen_vectors, 4)

    return pcs

