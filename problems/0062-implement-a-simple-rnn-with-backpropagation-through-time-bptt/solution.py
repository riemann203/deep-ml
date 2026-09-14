import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize random weights and zero column-vector biases."""
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """Cache states including h_0; return outputs of shape (T, d_y, 1)."""
        x = np.asarray(x, dtype=float)
        hidden_state = np.zeros((self.hidden_size, 1))
        self.hidden_states = [hidden_state]
        outputs = []

        for elem in x:
            elem = elem.reshape(-1, 1)
            hidden_state = np.tanh(
                self.W_xh @ elem + self.W_hh @ hidden_state + self.b_h
            )
            outputs.append(self.W_hy @ hidden_state + self.b_y)
            self.hidden_states.append(hidden_state)

        return np.asarray(outputs).reshape(len(x), self.W_hy.shape[0], 1)

    def backward(self, x, y, learning_rate):
        """Apply one BPTT update for half squared error summed over time/outputs."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError("inputs and targets must have the same sequence length")
        # Recompute the cache so backward also works without a prior forward call.
        outputs = self.forward(x)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        delta_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            h_prev = self.hidden_states[t]
            h = self.hidden_states[t + 1]
            error = outputs[t] - y[t].reshape(-1, 1)

            # Current output and future recurrence both contribute to dL/dh.
            dh = self.W_hy.T @ error + self.W_hh.T @ delta_next
            delta = dh * (1 - h ** 2)  # dL/da, where h = tanh(a)

            dW_hy += error @ h.T
            db_y += error
            dW_xh += delta @ x[t].reshape(1, -1)
            dW_hh += delta @ h_prev.T
            db_h += delta
            delta_next = delta

        # Shared parameters stay fixed until every time step has contributed.
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y