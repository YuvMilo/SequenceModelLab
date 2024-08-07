import torch
from typing import Callable, Tuple

from src.algorithms.misc import safe_complex_mm


def calc_kernel(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        ker_len: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
) -> torch.Tensor:
    x = torch.zeros([1, ker_len, 1]).to(A.device)
    x[0, 0, 0] = 1
    _, ker = recurrent_ssm_calculation(x, A, B, C, D, activation_function)
    ker = ker[0, :, 0]
    return ker


def recurrent_diag_ssm_calculation(
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        activation_function: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x is of shape B L in_D
    A is of shape H
    B is of shape N, in_D or N if input is 1D
    C is of shape N, out_D or N if output is 1D
    D is of shape out_D or 1 if output is 1D
    """

    batch_size, sequence_length, input_size = x.size()
    h = torch.zeros(A.shape[0], batch_size).to(A.device)

    A = A.view(-1, 1)
    D = D.view(-1, 1)

    if B.dim() == 1:
        B = B.view(-1, 1)

    if C.dim() == 1:
        C = C.view(1, -1)

    out = []
    hiddens = []
    for t in range(sequence_length):
        x_t = x[:, t, :]  # B, D
        x_t = x_t.transpose(0, 1)  # input_D, B
        h = safe_complex_mm(B, x_t) + h * A  # H, B
        h = activation_function(h)
        cur_out = safe_complex_mm(C, h) + D  # output_D, B
        out.append(cur_out.real)
        hiddens.append(h)

    out = torch.stack(out, dim=0)  # L, output_D ,B
    out = torch.permute(out, [2, 0, 1])  # B, L ,output_D

    hiddens = torch.stack(hiddens, dim=0)  # L, H, B
    hiddens = torch.permute(hiddens, [2, 0, 1])  # B, L, H

    return hiddens, out


def recurrent_ssm_calculation(
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        activation_function: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x is of shape B L in_D
    A is of shape H, H
    B is of shape N, in_D or N if input is 1D
    C is of shape N, out_D or N if output is 1D
    D is of shape out_D or 1 if output is 1D
    """

    batch_size, sequence_length, input_size = x.size()
    h = torch.zeros(A.shape[0], batch_size).to(A.device)

    D = D.view(-1, 1)

    if B.dim() == 1:
        B = B.view(-1, 1)

    if C.dim() == 1:
        C = C.view(1, -1)

    out = []
    hiddens = []
    for t in range(sequence_length):
        x_t = x[:, t, :]  # B, D
        x_t = x_t.transpose(0, 1)  # input_D, B
        h = safe_complex_mm(B, x_t) + safe_complex_mm(A, h)  # H, B
        h = activation_function(h)
        cur_out = safe_complex_mm(C, h) + D  # output_D, B
        out.append(cur_out.real)
        hiddens.append(h)

    out = torch.stack(out, dim=0)  # L, output_D ,B
    out = torch.permute(out, [2, 0, 1])  # B, L ,output_D

    hiddens = torch.stack(hiddens, dim=0)  # L, H, B
    hiddens = torch.permute(hiddens, [2, 0, 1])  # B, L, H

    return hiddens, out


def calc_kernel_diag(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        ker_len: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
) -> torch.Tensor:
    x = torch.zeros([1, ker_len, 1]).to(A.device)
    x[0, 0, 0] = 1
    _, ker = recurrent_diag_ssm_calculation(x, A, B, C, D, activation_function)
    ker = ker[0, :, 0]
    return ker
