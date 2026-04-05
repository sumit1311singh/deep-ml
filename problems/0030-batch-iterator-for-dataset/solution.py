import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n=X.shape[0]
	batches = []
	itr = 0
	while(itr<n):
		batch, y_batch = [], []
		for i in range(batch_size):
			if i+itr < n:
				batch.append(X[i+itr].tolist())
		#batches.append(batch)
		if y is not None:
			for i in range(batch_size):
				if i+itr < len(y) :
					y_batch.append(y[i+itr])
			#batches.append(y_batch)
		if len(y_batch):
			batches.append([batch, y_batch])
		else:
			batches.append(batch)
		itr+=batch_size
	return batches


