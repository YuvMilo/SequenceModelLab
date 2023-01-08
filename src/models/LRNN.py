import torch
import torch.nn as nn


class LinearRNN(nn.Module):

    def __init__(self, hidden_dim, input_size=1, output_size=1):
        super().__init__()
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_dim)*0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*0.001)
        self.b_ih = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hh = nn.Parameter(torch.zeros(hidden_dim))
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, output_size))
        self.b_ho = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()
        hidden_size = self.W_hh.size(0)

        # Initialize hidden state with zeros
        h = torch.zeros(batch_size, hidden_size)

        # Iterate over the time steps
        out = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = torch.mm(x_t, self.W_ih) + self.b_ih + torch.mm(h, self.W_hh) + self.b_hh
            cur_out = torch.mm(h, self.W_ho) + self.b_ho
            out.append(cur_out)

        out = torch.stack(out, dim=0)  # L, output_D ,B
        out = torch.permute(out, [1, 0, 2])  # B, L ,output_D

        return out
