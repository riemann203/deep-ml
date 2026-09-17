import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	def sigmoid(z: np.ndarray) -> np.ndarray:
		return 1.0 / (1.0 + np.exp(-z))

	features = np.asarray(features)
	weights = np.asarray(weights)
	labels = np.asarray(labels)
	probabilities = sigmoid(features @ weights + bias)
	mse = np.mean((probabilities - labels) ** 2)
	return np.round(probabilities, 4).tolist(), float(np.round(mse, 4))
