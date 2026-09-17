import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# z-score normalization
	means = np.mean(data, axis=0)
	stds = np.std(data, axis=0)
	standardized_data = np.divide(
		data-means,
		stds,
		out=np.zeros_like(data, dtype=float),
		where=(stds!=0.0)
	)

	# min-max normalization
	min_vals = np.min(data, axis=0)
	max_vals = np.max(data, axis=0)
	ranges = max_vals - min_vals
	normalized_data = np.divide(
		data-min_vals,
		ranges,
		out=np.zeros_like(data, dtype=float),
		where=(ranges!=0.0)
	)

	return np.round(standardized_data, 4), np.round(normalized_data, 4)