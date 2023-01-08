import torch

from src.models.strategies.base import BaseSMMCalcStrategy
from src.utils import safe_complex_mm


class RecurrentDiagSMMCalcStrategy(BaseSMMCalcStrategy):
    def __init__(self):
        super().__init__()

    def calc(self, x, A, B, C, D):
        """
        x is of shape B L in_D
        A is of shape H
        B is of shape N, in_D or N if input is 1D
        C is of shape N, out_D or N if output is 1D
        D is of shape out_D or 1 if output is 1D
        """

        batch_size, sequence_length, input_size = x.size()
        h = torch.zeros(A.shape[0], batch_size)

        A = A.view(-1, 1)
        D = D.view(-1, 1)

        if B.dim() == 1:
            B = B.view(-1, 1)

        if C.dim() == 1:
            C = C.view(1, -1)

        out = []
        for t in range(sequence_length):
            x_t = x[:, t, :]  # B, D
            x_t = x_t.transpose(0, 1)  # input_D, B
            h = safe_complex_mm(B, x_t) + h * A  # H, B
            cur_out = safe_complex_mm(C, h) + D  # output_D, B
            out.append(cur_out.real)

        out = torch.stack(out, dim=0)  # L, output_D ,B
        out = torch.permute(out, [2, 0, 1])  # B, L ,output_D

        return out


class RecurrentSMMCalcStrategy(BaseSMMCalcStrategy):
    def __init__(self):
        super().__init__()

    def calc(self, x, A, B, C, D):
        """
        x is of shape B L in_D
        A is of shape H, H
        B is of shape N, in_D or N if input is 1D
        C is of shape N, out_D or N if output is 1D
        D is of shape out_D or 1 if output is 1D
        """

        batch_size, sequence_length, input_size = x.size()
        h = torch.zeros(A.shape[0], batch_size)

        D = D.view(-1, 1)

        if B.dim() == 1:
            B = B.view(-1, 1)

        if C.dim() == 1:
            C = C.view(1, -1)

        out = []
        for t in range(sequence_length):
            x_t = x[:, t, :]  # B, D
            x_t = x_t.transpose(0, 1)  # input_D, B
            h = safe_complex_mm(B, x_t) + safe_complex_mm(A, h)  # H, B
            cur_out = safe_complex_mm(C, h) + D  # output_D, B
            out.append(cur_out.real)

        out = torch.stack(out, dim=0)  # L, output_D ,B
        out = torch.permute(out, [2, 0, 1])  # B, L ,output_D

        return out
