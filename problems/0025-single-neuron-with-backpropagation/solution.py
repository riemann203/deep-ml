import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-z))


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> Tuple[list[float], float, list[float]]:
	X = np.asarray(features, dtype=float)
	y = np.asarray(labels, dtype=float)

	w = np.asarray(initial_weights, dtype=float).copy()
	b = float(initial_bias)

	mse_values = []
	n = features.shape[0]

	for _ in range(epochs):
		outputs = sigmoid(X @ w + b)

		mse = np.mean((outputs - y) ** 2)
		mse_values.append(float(np.round(mse, 4)))

		delta = 2 * (outputs - y) * outputs * (1.0 - outputs)

		grad_w = X.T @ delta / n
		grad_b = np.mean(delta)

		w -= learning_rate * grad_w
		b -= learning_rate * grad_b

	return np.round(w, 4).tolist(), round(b, 4), mse_values