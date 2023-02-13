import torch
import torch.nn as nn


class SMMModel(nn.Module):

    def __init__(self, ssm_param_strategy, ssm_calc_strategy, num_hidden_state,
                 input_dim, output_dim, trainable_param_list, device):
        super().__init__()

        self.ssm_param_strategy = ssm_param_strategy
        self.ssm_calc_strategy = ssm_calc_strategy
        self.num_hidden_state = num_hidden_state
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trainable_param_list = trainable_param_list

        # TODO - This should be a "running context"
        # Should be refactored to be a class
        self.device = device
        p_A, p_B, p_C, p_D = ssm_param_strategy.init_param(
            num_hidden_state=num_hidden_state,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.device
        )

        self.parameterized_A = nn.Parameter(p_A.to(device),
                                            requires_grad="A" in
                                                          self.trainable_param_list)
        self.parameterized_B = nn.Parameter(p_B.to(device),
                                            requires_grad="B" in
                                                          self.trainable_param_list)
        self.parameterized_C = nn.Parameter(p_C.to(device),
                                            requires_grad="C" in
                                                          self.trainable_param_list)
        self.parameterized_D = nn.Parameter(p_D.to(device),
                                            requires_grad="D" in
                                                          self.trainable_param_list)

    def forward(self, x):
        A, B, C, D = self.ssm_param_strategy.get_param(self.parameterized_A,
                                                       self.parameterized_B,
                                                       self.parameterized_C,
                                                       self.parameterized_D,
                                                       self.device)

        out = self.ssm_calc_strategy.calc(x, A, B, C, D)
        return out

    def to(self, device):
        self.device = device
        return super()

    def get_kernel(self, ker_len):
        if self.input_dim != 1 or self.output_dim != 1:
            raise NotImplementedError("get_kernel are only implemented for 1D to 1D")
        # This could be done more efficiently in the not 1D case
        x = torch.zeros([1, ker_len, 1])
        x[0, 0, 0] = 1
        x = x.to(self.device)
        ker = self.forward(x)
        ker = ker[0, :, 0]
        return ker

    def get_params(self):
        A, B, C, D = self.ssm_param_strategy.get_param(self.parameterized_A,
                                                       self.parameterized_B,
                                                       self.parameterized_C,
                                                       self.parameterized_D,
                                                       device=self.device)
        return torch.clone(A), torch.clone(B), torch.clone(C), torch.clone(D)

    def get_num_hidden_state(self):
        return self.num_hidden_state

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim
