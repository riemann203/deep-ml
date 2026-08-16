import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1]
	determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

	leading_coefficient = 1.0
	discriminant = trace ** 2 - 4 * leading_coefficient * determinant
	eigenvalues = [
		(trace + eps * math.sqrt(discriminant)) / (2 * leading_coefficient)
		for eps in [-1.0, 1.0]
	]
	if discriminant > 0:
		eigenvalues = sorted(eigenvalues, reverse=True)

	return eigenvalues