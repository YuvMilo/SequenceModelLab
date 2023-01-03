import math
import torch
import torch.nn as nn
import torch.fft
from einops import repeat


_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class LS4D(nn.Module):


    def __init__(self, d_model, N=64, dt=0.01, L = 100):
        super().__init__()
        # Generate dt
        H = d_model

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register_parameter("log_A_real", nn.Parameter(log_A_real))
        self.register_parameter("A_imag", nn.Parameter(A_imag))

        self.L = L
        self.dt = dt

    def forward(self, x):
        batch_size, sequence_length, input_size = x.size()

        dt = self.dt
        C = _r2c(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        # At = A/dt
        # Ct = C/dt

        A = ((torch.eye(A.shape[0]) - A/2*dt)**-1)*(torch.eye(A.shape[0]) + A/2*dt)
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

class LS4DSimplified(nn.Module):


    def __init__(self, d_model, N=64, dt=0.01, lr=None, L = 100):
        super().__init__()
        # Generate dt
        H = d_model

        A_real = 0.5 * torch.ones(H, N // 2)
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
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

