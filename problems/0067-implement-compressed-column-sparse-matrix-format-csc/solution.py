def compressed_col_sparse_matrix(dense_matrix: list[list[float]]):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	vals = []
	row_indices = []
	col_pointers = [0]
	num_rows, num_cols = len(dense_matrix), len(dense_matrix[0])

	for col_idx in range(num_cols):
		for row_idx in range(num_rows):
			val = dense_matrix[row_idx][col_idx]
			if val != 0.0:
				vals.append(val)
				row_indices.append(row_idx)
		col_pointers.append(len(vals))
	return vals, row_indices, col_pointers