import numpy as np

def shuffle_data(X, y, seed=None):
	if seed is not None:
		np.random.seed(seed)

	num_samples = X.shape[0]
	indices = np.arange(num_samples)

	np.random.shuffle(indices)

	return X[indices, :], y[indices]