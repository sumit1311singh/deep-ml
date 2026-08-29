import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    k = len(mu_p)
    Cov_q_inv = np.linalg.inv(Cov_q)
    Cov_q_inv_cov_p = Cov_q_inv @ Cov_p
    tr = np.trace(Cov_q_inv_cov_p)
    mean_diff = mu_q-mu_p
    quad_form = (mean_diff.T @ Cov_q_inv) @ mean_diff
    log_val = np.log(np.linalg.det(Cov_q)/np.linalg.det(Cov_p))
    return 1/2*(tr + quad_form + log_val - k)
