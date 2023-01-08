from src.models.strategies.base import BaseSSMDiscretizationStrategy
from src.utils import safe_complex_mm


class BilinearDiagSSMDiscretizationStrategy(BaseSSMDiscretizationStrategy):

    def __init__(self):
        super().__init__()

    def discretize(self, dt, A, B, C, D):
        left_disc_const = (1 - A / 2 * dt) ** -1
        A = left_disc_const * (1 + A / 2 * dt)
        C = left_disc_const * (C * dt)
        return A, B, C, D


class BilinearSSMDiscretizationStrategy(BaseSSMDiscretizationStrategy):

    def __init__(self):
        super().__init__()

    def discretize(self, dt, A, B, C, D):
        left_disc_const = (1 - A / 2 * dt).inverse()
        A = safe_complex_mm(left_disc_const, (1 + A / 2 * dt))
        C = safe_complex_mm((C * dt), left_disc_const)
        return A, B, C, D
