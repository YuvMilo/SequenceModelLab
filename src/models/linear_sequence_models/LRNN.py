import torch
import torch.nn as nn


class OneLayerDiagTransitionLinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 std_Wih=1e-4, std_Whh1=1e-4):
        super().__init__()
        self.b = nn.Parameter(torch.randn(input_size, hidden_size) * (std_Wih ** 2))
        self.A = nn.Parameter(torch.randn(hidden_size) * (std_Whh1 ** 2))
        self.c = nn.Parameter(torch.randn(hidden_size, output_size))

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()
        hidden_size = self.A.size(0)

        # Initialize hidden state with zeros
        h = torch.zeros(batch_size, hidden_size)

        # Iterate over the time steps
        out = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = torch.mm(x_t, self.b) + h * self.A
            out.append(torch.mm(h, self.c))

        out = torch.concat(out, dim=1)
        if len(out.shape) == 2:
            out = torch.unsqueeze(out, dim=-1)

        return out
