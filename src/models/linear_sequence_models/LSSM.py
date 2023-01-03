import math
import torch
import torch.nn as nn
import torch.fft
from einops import repeat


_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class LSSMWithS4Parameterization(nn.Module):


    def __init__(self, d_model, N=64, dt=0.01, lr=None, L = 100):
        super().__init__()
        # Generate dt
        H = d_model

        A_real = -0.2 * torch.ones(H, N // 2) + torch.randn(H, N // 2) * 0.02
        A_imag = math.pi * torch.randn(H, N // 2) * 4

        A = A_real + 1j * A_imag
        self.A = nn.Parameter(_c2r(A))

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))

        self.L = L
        self.dt = dt

    def forward(self, x):
        batch_size, sequence_length, input_size = x.size()

        dt = self.dt
        C = _r2c(self.C)
        A = _r2c(self.A)

        A = ((torch.eye(A.shape[0]) - A / 2 * dt) ** -1) * (torch.eye(A.shape[0]) + A / 2 * dt)
        C = ((torch.eye(A.shape[0]) - A/2*dt)**-1) * C * dt

        h = torch.zeros(batch_size, A.shape[0])

        # Iterate over the time steps
        out = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = x_t + h * A
            out.append(torch.mm(h, C.transpose(0, 1)).real)

        out = torch.concat(out, dim=1)
        if len(out.shape) == 2:
            out = torch.unsqueeze(out, dim=-1)

        return out

