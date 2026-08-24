import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    X=np.array(X)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    X=X/norm
    return np.round(np.dot(X, X.T), 4)