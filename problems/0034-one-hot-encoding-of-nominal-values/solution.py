import numpy as np

def to_categorical(x, n_col=None):
    x = np.array(x)
    
    # If n_col not provided, infer from unique values
    if n_col is None:
        n_col = len(np.unique(x))
    
    # Decide whether to build a mapping
    if x.dtype.kind in {'i', 'u'} and x.min() >= 0 and x.max() < n_col:
        # Integers already in valid range [0, n_col-1]
        mappingDict = {val: val for val in np.unique(x)}
    else:
        # Build mapping for strings or arbitrary integers
        mappingDict = {val: idx for idx, val in enumerate(np.unique(x))}
    
    ans = np.zeros((len(x), n_col))
    for i in range(len(x)):
        ans[i][mappingDict[x[i]]] = 1
    return ans
