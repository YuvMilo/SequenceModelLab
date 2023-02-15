import numpy as np
import torch

from src.algorithms.ssm_calc import calc_kernel


INSIGNIFICANT_LOSS_CHANGE = 0.02


def effective_train_len_augmentation(epochs, losses):
    min_loss = np.min(losses)
    index_of_effective_train_finished = np.where(
        losses < INSIGNIFICANT_LOSS_CHANGE + min_loss
    )[0][0]
    return epochs[index_of_effective_train_finished]


def effective_min_loss_augmentation(epochs, losses):
    min_loss = np.min(losses)
    return min_loss


def eig_augmentation(matrix):
    return np.linalg.eig(matrix)[0]


def svd_augmentation(matrix):
    return np.linalg.svd(matrix)[1]


def hankel_augmentation(A, B, C):
    dim = A.shape[0]
    curr_impulse_response = calc_kernel(A=torch.Tensor(A),
                                        B=torch.Tensor(B),
                                        C=torch.Tensor(C),
                                        D=torch.Tensor([0]),
                                        ker_len=2 * dim)
    hankel_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            hankel_matrix[i, j] = curr_impulse_response[i + j]
    return hankel_matrix
