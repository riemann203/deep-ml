import numpy as np

def to_categorical(x, n_col=None):
	if n_col is None:
		n_col = np.max(x) + 1
	one_hot = np.zeros(shape=(len(x), n_col))
	one_hot[np.arange(len(x)), x] = 1.0
	return one_hot