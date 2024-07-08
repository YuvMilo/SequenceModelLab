import torch
import numpy as np
from collections import defaultdict


def safe_complex_mm(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    A matrix multiplication function that works if either of m1 or m2 are complex
    (current torch.mm works only if m1 and m2 are of the same type)
    """
    if m1.type() == m2.type():
        return torch.mm(m1, m2)
    else:
        m1 = m1.type(torch.cfloat)
        m2 = m2.type(torch.cfloat)
        return torch.mm(m1, m2)


def polar_to_complex(radii, angles_radians):
    return radii * np.exp(1j * angles_radians)


def get_2x2matrix_with_eigen(eigen: float,
                             main_diagonal_diff: float = 0,
                             off_diagonal_ratio: float = 1):
    a = eigen.real + main_diagonal_diff / 2
    d = eigen.real - main_diagonal_diff / 2

    b_times_c_abs = (eigen.imag ** 2) + ((main_diagonal_diff ** 2) / 4)
    b = - 1 * (b_times_c_abs ** 0.5) * (off_diagonal_ratio ** 0.5)
    c = (b_times_c_abs ** 0.5) / (off_diagonal_ratio ** 0.5)

    rot = torch.zeros([2, 2])
    # this matrix has eigenvalue get_real_i(i) +- i*get_imag_i(i)
    rot[0, 0] = a
    rot[1, 1] = d
    rot[0, 1] = b
    rot[1, 0] = c
    return rot


def matrix_to_real_2x2block_matrix_with_same_eigenvalues(matrix):
    eig, v = torch.linalg.eig(matrix)

    normalized_eig_to_count = defaultdict(int)
    real_eigs = []
    for e in eig:
        if e.imag == 0:
            real_eigs.append(e)
        else:
            e.imag = torch.abs(e.imag)
            normalized_eig_to_count[complex(e)] += 1

    imag_eigs = []
    for e in normalized_eig_to_count.keys():
        for _ in range(normalized_eig_to_count[e] // 2):
            imag_eigs.append(e)

    A = torch.zeros(matrix.shape)
    for i in range(0, len(imag_eigs) * 2, 2):
        A[i:i + 2, i:i + 2] = get_2x2matrix_with_eigen(imag_eigs[i // 2])

    A[len(imag_eigs) * 2:, len(imag_eigs) * 2:] = torch.diag(torch.Tensor(real_eigs))

    A.to(matrix.dtype)
    return A
