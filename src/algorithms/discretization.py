from typing import List

import torch

from src.algorithms.misc import safe_complex_mm


# TODO - Write test for this
def bilinear_diag_discretization(dt: float, A: torch.Tensor,
                                 B: torch.Tensor,
                                 C: torch.Tensor,
                                 D: torch.Tensor) -> List[torch.Tensor]:
    left_disc_const = (1 - A / 2 * dt) ** -1
    A = left_disc_const * (1 + A / 2 * dt)
    C = left_disc_const * (C * dt)
    return A, B, C, D


# TODO - Write test for this
def bilinear_discretization(dt: float, A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            D: torch.Tensor) -> List[torch.Tensor]:
    eye = torch.eye(A.shape[0]).to(A.device)
    left_disc_const = (eye - A / 2 * dt).inverse()
    A = safe_complex_mm(left_disc_const, (eye + A / 2 * dt))
    C = safe_complex_mm((C * dt), left_disc_const)
    return A, B, C, D
