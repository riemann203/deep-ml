import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(
            torch.empty(4 * hidden_size, input_size)
        )
        self.W_hh = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size)
        )
        self.b_ih = nn.Parameter(
            torch.empty(4 * hidden_size)
        )
        self.b_hh = nn.Parameter(
            torch.empty(4 * hidden_size)
        )

        k = 1 / math.sqrt(hidden_size)
        nn.init.uniform_(self.W_ih, -k, k)
        nn.init.uniform_(self.W_hh, -k, k)
        nn.init.uniform_(self.b_ih, -k, k)
        nn.init.uniform_(self.b_hh, -k, k)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        h_prev, c_prev = state
        gates = (
            x @ self.W_ih.T
            + self.b_ih
            + h_prev @ self.W_hh.T
            + self.b_hh
        )
        input_gate, forget_gate, candidate_cell, output_gate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate_cell = torch.tanh(candidate_cell)
        output_gate = torch.sigmoid(output_gate)

        c_new = forget_gate * c_prev + input_gate * candidate_cell
        h_new = output_gate * torch.tanh(c_new)
        return h_new, c_new