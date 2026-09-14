import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	Wx = np.asarray(Wx)
	Wh = np.asarray(Wh)
	b = np.asarray(b)
	hidden_state = np.asarray(initial_hidden_state)
	for x in input_sequence:
		x = np.asarray(x)
		hidden_state = np.tanh(Wx @ x + Wh @ hidden_state + b)

	return np.round(hidden_state, 4).tolist()
