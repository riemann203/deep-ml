import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
	return np.where(z>0, z, 0)


def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	w1 = np.asarray(w1, dtype=float)
	w2 = np.asarray(w2, dtype=float)

	return relu(x + w2 @ relu(w1 @ x))