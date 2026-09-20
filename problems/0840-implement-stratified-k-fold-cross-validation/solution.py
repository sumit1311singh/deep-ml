import numpy as np

def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	y = np.array(y)

	unique_classes, counts = np.unique(y, return_counts=True)

	num_classes = len(unique_classes)

	class_to_indices = {}
	for cls in unique_classes:
		class_to_indices[cls] = np.where(y == cls)[0]

	groups_per_class = {}
	for cls in unique_classes:
		indices = class_to_indices[cls]
		size = len(indices)
		base = size//n_splits
		extra = size%n_splits

		groups = []
		start = 0
		for i in range(n_splits):
			group_size = base + (1 if i<extra else 0)
			groups.append(indices[start:start+group_size])
			start+=group_size
		
		groups_per_class[cls] = groups
	
	folds = []

	for i in range(n_splits):
		test_indices = np.concatenate([groups_per_class[cls][i] for cls in unique_classes])
		train_indices = np.concatenate([
				np.concatenate(groups_per_class[cls][:i] + groups_per_class[cls][i+1:])
				for cls in unique_classes
			])
		
		test_indices = np.sort(test_indices)
		train_indices = np.sort(train_indices)
		folds.append([train_indices.tolist(), test_indices.tolist()])

	return folds

