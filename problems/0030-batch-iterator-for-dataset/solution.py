import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	num_samples = X.shape[0]
	for i in range(0, num_samples, batch_size):
		if y is None:
			yield [X[i:i+batch_size, :].tolist()]
		else:
			yield [X[i:i+batch_size, :].tolist(), y[i:i+batch_size].tolist()]