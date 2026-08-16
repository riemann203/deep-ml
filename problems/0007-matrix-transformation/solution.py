import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.asarray(A, dtype=float)
	T = np.asarray(T, dtype=float)
	S = np.asarray(S, dtype=float)

	det_T = np.linalg.det(T)
	det_S = np.linalg.det(S)

	if det_T == 0.0 or det_S == 0.0:
		return -1

	inv_T = np.linalg.inv(T)

	return inv_T @ A @ S