import math

import torch

from src.models.strategies.base import BaseSSMInitStrategy


class DiagHippoInitStrategy(BaseSSMInitStrategy):

    def __init__(self):
        pass

    def get_init_params(self, num_hidden_state,
                        input_dim, output_dim):
        if input_dim != 1 or output_dim != 1:
            raise NotImplementedError(
                "currently DiagHippoInitStrategy is only implemented in"
                "1D to 1D dynamics"
            )

        num_hidden_state_in_practice = num_hidden_state // 2

        A_real = -0.5 * torch.ones(num_hidden_state_in_practice)
        A_imag = math.pi * torch.arange(num_hidden_state_in_practice)
        A = A_real + 1j * A_imag

        B = torch.ones([num_hidden_state_in_practice, input_dim], dtype=torch.cfloat)
        C = torch.randn(output_dim, num_hidden_state_in_practice, dtype=torch.cfloat)

        D = torch.zeros([output_dim, 1], dtype=torch.cfloat)

        return A, B, C, D


class FlexibleHippoInitStrategy(BaseSSMInitStrategy):

    def __init__(self, A_init_func, B_init_func, C_init_func, D_init_func):
        self.A_init_func = A_init_func
        self.B_init_func = B_init_func
        self.C_init_func = C_init_func
        self.D_init_func = D_init_func

    def get_init_params(self, num_hidden_state,
                        input_dim, output_dim):
        if input_dim != 1 or output_dim != 1:
            raise NotImplementedError(
                "currently DiagHippoInitStrategy is only implemented in"
                "1D to 1D dynamics"
            )

        return self.A_init_func(num_hidden_state // 2), \
            self.B_init_func(num_hidden_state // 2), \
            self.C_init_func(num_hidden_state // 2), \
            self.D_init_func(num_hidden_state // 2)


class FlexibleInitStrategy(BaseSSMInitStrategy):

    def __init__(self, A_init_func, B_init_func, C_init_func, D_init_func):
        self.A_init_func = A_init_func
        self.B_init_func = B_init_func
        self.C_init_func = C_init_func
        self.D_init_func = D_init_func

    def get_init_params(self, num_hidden_state,
                        input_dim, output_dim):
        if input_dim != 1 or output_dim != 1:
            raise NotImplementedError(
                "currently DiagHippoInitStrategy is only implemented in"
                "1D to 1D dynamics"
            )

        return self.A_init_func(num_hidden_state), \
            self.B_init_func(num_hidden_state), \
            self.C_init_func(num_hidden_state), \
            self.D_init_func(num_hidden_state)
