import torch

from src.models.strategies.base import BaseSSMStoringStrategy


class ComplexAs2DRealArrayStoringStrategy(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        A = torch.view_as_real(A)
        B = torch.view_as_real(B)
        C = torch.view_as_real(C)
        D = torch.view_as_real(D)
        return A, B, C, D

    def load(self, A, B, C, D):
        A = torch.view_as_complex(A)
        B = torch.view_as_complex(B)
        C = torch.view_as_complex(C)
        D = torch.view_as_complex(D)
        return A, B, C, D


class RealArrayStoringStrategy(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        return A, B, C, D

    def load(self, A, B, C, D):
        return A, B, C, D
