import numpy as np

def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    dictt = {i: 0.0 for i in range(n_features)}
    for tree in trees:
        for node in tree:
            if node['feature_index'] in dictt:
                dictt[node['feature_index']] += node['impurity_decrease']
            else:
                dictt[node['feature_index']] = node['impurity_decrease']
    result = np.array([dictt[i] for i in range(n_features)])
    if sum(result)!=0:
        result = result/sum(result)
    #print("Result: ", result, "\n\n")
    return result.tolist()
    