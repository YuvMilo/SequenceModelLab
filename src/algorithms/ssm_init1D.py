import numpy as np
import torch
from typing import List, Callable

from src.algorithms.misc import polar_to_complex
from src.algorithms.discretization import bilinear_discretization
from src.algorithms.misc import get_2x2matrix_with_eigen


def get_hippo_cont_init(num_hidden_state: int,
                        C_init_std: int = 1,
                        input_dim: int = 1,
                        output_dim: int = 1) -> List[torch.Tensor]:
    if input_dim != 1 or output_dim != 1:
        raise NotImplementedError("get_hippo_cont only works with input"
                                  "and output dim 1")

    q = np.arange(num_hidden_state, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    A = torch.Tensor(A)

    B = np.diag(T)[:, None]
    B = torch.Tensor(B.copy())

    C = C_init_std * torch.randn([1, num_hidden_state], dtype=torch.float)
    D = torch.zeros([1, 1], dtype=torch.float)

    return A, B, C, D


def get_hippo_disc_init(num_hidden_state: int,
                        C_init_std: float = 1,
                        dt: float = 0.01,
                        input_dim: int = 1,
                        output_dim: int = 1) -> List[torch.Tensor]:
    A, B, C, D = get_hippo_cont_init(num_hidden_state=num_hidden_state,
                                     C_init_std=C_init_std,
                                     input_dim=input_dim,
                                     output_dim=output_dim)

    A, B, C, D = bilinear_discretization(dt=dt, A=A, B=B,
                                         C=C, D=D)

    return A, B, C, D


def get_diag_ssm_plus_noise_init(num_hidden_state: int,
                                 A_diag: float = 0.9,
                                 A_noise_std: float = 0.001,
                                 B_init_std: float = 1e-1,
                                 C_init_std: float = 1e-1,
                                 input_dim: int = 1,
                                 output_dim: int = 1):
    A = A_diag * torch.eye(num_hidden_state) + \
        torch.randn([num_hidden_state, num_hidden_state]) * A_noise_std
    B = B_init_std * torch.randn([num_hidden_state, input_dim], dtype=torch.float)
    C = C_init_std * torch.randn([output_dim, num_hidden_state], dtype=torch.float)
    D = torch.zeros([output_dim, 1], dtype=torch.float)

    return A, B, C, D


def get_rot_ssm_matrix(num_hidden_state: int,
                       get_i_eigen: Callable[[int], complex],
                       main_diagonal_diff: float = 0,
                       off_diagonal_ratio: float = 1):
    A = torch.zeros([num_hidden_state, num_hidden_state])
    for i in range(0, num_hidden_state, 2):
        A[i:i + 2, i:i + 2] = get_2x2matrix_with_eigen(get_i_eigen(i // 2),
                                                       main_diagonal_diff=main_diagonal_diff,
                                                       off_diagonal_ratio=off_diagonal_ratio)
    return A


def get_rot_ssm_equally_spaced_init(num_hidden_state: int,
                                    radii: float = 0.99,
                                    B_init_std: float = 1e-1,
                                    C_init_std: float = 1e-1,
                                    angle_shift: float = 2 ** 0.5,
                                    main_diagonal_diff: float = 0,
                                    off_diagonal_ratio: float = 1,
                                    input_dim: int = 1,
                                    output_dim: int = 1):
    eff_hidden = num_hidden_state // 2

    def get_i_eigen(i: int) -> complex:
        angle = np.pi * (i / (eff_hidden - 1))
        angle += angle_shift
        angle = angle % (np.pi + 0.0001)
        return polar_to_complex(radii, angle)

    A = get_rot_ssm_matrix(num_hidden_state=num_hidden_state,
                           get_i_eigen=get_i_eigen,
                           off_diagonal_ratio=off_diagonal_ratio,
                           main_diagonal_diff=main_diagonal_diff)
    B = B_init_std * torch.randn([num_hidden_state, input_dim], dtype=torch.float)
    C = C_init_std * torch.randn([output_dim, num_hidden_state], dtype=torch.float)
    D = torch.zeros([output_dim, 1], dtype=torch.float)

    return A, B, C, D


def get_rot_ssm_one_over_n_init(num_hidden_state: int,
                                radii: float = 0.99,
                                B_init_std: float = 1e-1,
                                C_init_std: float = 1e-1,
                                main_diagonal_diff: float = 0,
                                off_diagonal_ratio: float = 1,
                                input_dim: int = 1,
                                output_dim: int = 1):
    def get_i_eigen(i: int) -> complex:
        return polar_to_complex(radii, 2 * np.pi / (i + 1))

    A = get_rot_ssm_matrix(num_hidden_state=num_hidden_state,
                           get_i_eigen=get_i_eigen,
                           off_diagonal_ratio=off_diagonal_ratio,
                           main_diagonal_diff=main_diagonal_diff)
    B = B_init_std * torch.randn([num_hidden_state, input_dim], dtype=torch.float)
    C = C_init_std * torch.randn([output_dim, num_hidden_state], dtype=torch.float)
    D = torch.zeros([output_dim, 1], dtype=torch.float)

    return A, B, C, D
