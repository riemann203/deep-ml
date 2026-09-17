import numpy as np


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B = np.asarray(B, dtype=float)
	C = np.asarray(C, dtype=float)
	return (np.linalg.inv(C) @ B).tolist()
