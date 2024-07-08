from src.models.strategies.base import BaseSSMDiscretizationStrategy
from src.algorithms.discretization import bilinear_diag_discretization, \
    bilinear_discretization


class BilinearDiagSSMDiscretizationStrategy(BaseSSMDiscretizationStrategy):

    def __init__(self):
        super().__init__()

    def discretize(self, dt, A, B, C, D):
        return bilinear_diag_discretization(dt, A, B, C, D)


class BilinearSSMDiscretizationStrategy(BaseSSMDiscretizationStrategy):

    def __init__(self):
        super().__init__()

    def discretize(self, dt, A, B, C, D, device):
        return bilinear_discretization(dt, A, B, C, D)
