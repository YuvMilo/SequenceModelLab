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


class BandComplexStoring(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        B = torch.view_as_real(B)
        C = torch.view_as_real(C)
        D = torch.view_as_real(D)
        return A, B, C, D

    def load(self, A, B, C, D):
        band_angle = A
        n = len(B)
        angles = torch.linspace(-1, 1, n,device=A.device) * band_angle
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)
        A = torch.complex(real_part, imag_part)

        B = torch.view_as_complex(B)
        C = torch.view_as_complex(C)
        D = torch.view_as_complex(D)
        return A, B, C, D


class ComplexAs2DRealArrayStoringStrategyBoundedByOne(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        A_norms = torch.abs(A)
        A = A / A_norms
        A = A * torch.min(A_norms, torch.ones_like(A_norms) * 0.999)
        A = torch.view_as_real(A)
        B = torch.view_as_real(B)
        C = torch.view_as_real(C)
        D = torch.view_as_real(D)
        return A, B, C, D

    def load(self, A, B, C, D):
        A = torch.view_as_complex(A)
        A_norms = torch.abs(A)
        A = A / A_norms
        A = A * torch.min(A_norms, torch.ones_like(A_norms) * 0.999)
        B = torch.view_as_complex(B)
        C = torch.view_as_complex(C)
        D = torch.view_as_complex(D)
        return A, B, C, D


class ComplexAsPolarBoundedByOne(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        angles = torch.angle(A)
        radiis = torch.abs(A)
        log_radiis = torch.log(radiis)
        log_minus_log_radiis = torch.log(-log_radiis)
        stored_A = torch.concatenate([angles, log_minus_log_radiis])
        B = torch.view_as_real(B)
        C = torch.view_as_real(C)
        D = torch.view_as_real(D)
        return stored_A, B, C, D

    def load(self, A, B, C, D):
        stored_A = A
        loaded_angles, loaded_log_minus_log_radiis = torch.split(stored_A, len(stored_A) // 2)
        loaded_A = torch.exp(-1 * torch.exp(loaded_log_minus_log_radiis) + 1j * loaded_angles)

        B = torch.view_as_complex(B)
        C = torch.view_as_complex(C)
        D = torch.view_as_complex(D)
        return loaded_A, B, C, D



class ComplexAs2DRealArrayStoringStrategyOnlyA(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        A = torch.view_as_real(A)
        return A, B, C, D

    def load(self, A, B, C, D):
        A = torch.view_as_complex(A)
        return A, B, C, D


class RealArrayStoringStrategy(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        return A, B, C, D

    def load(self, A, B, C, D):
        return A, B, C, D


class RealArrayStoringStrategyBoudedByOne(BaseSSMStoringStrategy):

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        A_norms = torch.abs(A)
        A = A / A_norms
        A = A * torch.min(A_norms, torch.ones_like(A_norms) * 0.999)
        return A, B, C, D

    def load(self, A, B, C, D):
        A_norms = torch.abs(A)
        A = A / A_norms
        A = A * torch.min(A_norms, torch.ones_like(A_norms)*0.999)
        return A, B, C, D


class BlockDiagStoringStrategy(BaseSSMStoringStrategy):
    def __init__(self):
        self.mask = None

    def store(self, A, B, C, D):
        matrix2x2 = torch.tensor([[1, 1], [1, 1]])
        matrix1x1 = torch.tensor([[1]])
        self.mask = torch.block_diag(*([matrix2x2] * (A.shape[0] // 2)
                                       + [matrix1x1] * (A.shape[0] % 2)))

        return A, B, C, D

    def load(self, A, B, C, D):
        if self.mask is not None:
            return A * self.mask.to(A.device), B, C, D
        return A, B, C, D
