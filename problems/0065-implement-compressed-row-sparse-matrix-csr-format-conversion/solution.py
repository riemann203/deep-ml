import numpy as np

def compressed_row_sparse_matrix(dense_matrix: list[list[float]]):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	vals = []
	col_indices = []
	row_pointers = [0]
	for row in dense_matrix:
		for j, val in enumerate(row):	
			if val != 0.0:
				vals.append(val)
				col_indices.append(j)
		row_pointers.append(len(vals))
	return vals, col_indices, row_pointers