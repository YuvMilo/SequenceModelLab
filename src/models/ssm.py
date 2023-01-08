import torch.nn as nn


class SMMModel(nn.Module):

    def __init__(self, ssm_param_strategy, ssm_calc_strategy, num_hidden_state,
                 input_dim, output_dim, trainable_param_list):
        super().__init__()

        self.ssm_param_strategy = ssm_param_strategy
        self.ssm_calc_strategy = ssm_calc_strategy
        self.num_hidden_state = num_hidden_state
        p_A, p_B, p_C, p_D = ssm_param_strategy.init_param(
            num_hidden_state=num_hidden_state,
            input_dim=input_dim,
            output_dim=output_dim
        )

        self.parameterized_A = nn.Parameter(p_A,
                                            requires_grad="A" in trainable_param_list)
        self.parameterized_B = nn.Parameter(p_B,
                                            requires_grad="B" in trainable_param_list)
        self.parameterized_C = nn.Parameter(p_C,
                                            requires_grad="C" in trainable_param_list)
        self.parameterized_D = nn.Parameter(p_D,
                                            requires_grad="D" in trainable_param_list)

    def forward(self, x):
        A, B, C, D = self.ssm_param_strategy.get_param(self.parameterized_A,
                                                       self.parameterized_B,
                                                       self.parameterized_C,
                                                       self.parameterized_D)

        out = self.ssm_calc_strategy.calc(x, A, B, C, D)
        return out
