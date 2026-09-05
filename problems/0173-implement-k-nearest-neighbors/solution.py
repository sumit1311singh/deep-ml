import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.array(points)

    distances = []
    distances = np.sqrt(np.sum((points - query_point)**2, axis=1))

    idx = np.argpartition(distances, k-1)[:k]
    idx = idx[np.argsort(distances[idx], kind='stable')]

    return [tuple(points[i]) for i in idx]