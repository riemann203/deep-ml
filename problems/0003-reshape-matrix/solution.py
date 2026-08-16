import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	num_rows = len(a)
	num_cols = len(a[0])

	if num_rows * num_cols != new_shape[0] * new_shape[1]:
		return []

	reshaped_matrix = [[] for _ in range(new_shape[0])]
	for i, row in enumerate(a):
		for j, elem in enumerate(row):
			idx = i * num_cols + j
			col_idx = idx // new_shape[1]
			reshaped_matrix[col_idx].append(elem)

	return reshaped_matrix