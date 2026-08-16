def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	if not vectors:
		return []

	means = [
		sum(elem for elem in vector) / len(vector) for vector in vectors
	]
	centered = [
		[elem - mean for elem in vector]
		for mean, vector in zip(means, vectors)
	]

	num_obs_minus_1 = len(vectors[0]) - 1
	covariance_matrix = [[0.0 for _ in range(len(vectors))] for _ in range(len(vectors))]
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i <= j:
				covariance_matrix[i][j] = sum(
					elem1 * elem2 for elem1, elem2 in zip(centered[i], centered[j])
				) / (num_obs_minus_1)
			else:
				covariance_matrix[i][j] = covariance_matrix[j][i]

	return covariance_matrix