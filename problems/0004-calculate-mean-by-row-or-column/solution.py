def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if len(matrix) == 0:
		return []

	num_rows = len(matrix)
	num_cols = len(matrix[0])

	if mode == "row":
		return [
			sum(matrix[i][j] for j in range(num_cols)) / num_cols
			for i in range(num_rows)
		]
	elif mode == "column":
		return [
			sum(matrix[i][j] for i in range(num_rows)) / num_rows
			for j in range(num_cols)
		]