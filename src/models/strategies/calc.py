from src.models.strategies.base import BaseSMMCalcStrategy
from src.algorithms.ssm import recurrent_diag_ssm_calculation,\
    recurrent_ssm_calculation


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

        return recurrent_diag_ssm_calculation(x, A, B, C, D)


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

        return recurrent_ssm_calculation(x, A, B, C, D)
