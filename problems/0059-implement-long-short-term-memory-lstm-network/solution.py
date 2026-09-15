import numpy as np

def sigmoid(x: float):
	return 1.0 / (1.0 + np.exp(-x))

class LSTM:
	def __init__(self, input_size: int, hidden_size: int):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x: np.ndarray, initial_hidden_state: np.ndarray, initial_cell_state: np.ndarray):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		x = np.asarray(x)
		hidden_states = [initial_hidden_state.reshape(-1, 1)]
		cell_state = initial_cell_state.reshape(-1, 1)
		for elem in x:
			elem = elem.reshape(-1, 1)
			vec = np.concatenate([hidden_states[-1], elem])
			f = sigmoid(self.Wf @ vec + self.bf)
			i = sigmoid(self.Wi @ vec + self.bi)
			o = sigmoid(self.Wo @ vec + self.bo)
			candidate_memory_cell = np.tanh(self.Wc @ vec + self.bc)
			cell_state = f * cell_state + i * candidate_memory_cell
			hidden_states.append(o * np.tanh(cell_state))
		return hidden_states, hidden_states[-1], cell_state