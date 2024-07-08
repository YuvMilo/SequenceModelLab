import torch
from torch import nn

from src.models.ssm import SMMModel


class DeepSMMModel(nn.Module):

    def __init__(self, ssm_param_strategy, ssm_calc_strategy, num_hidden_state,
                 input_dim, output_dim, trainable_param_list, device, depth,
                 non_linearity=lambda x: x):
        super().__init__()

        self.num_hidden_state = num_hidden_state
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth

        self.layers = []
        first = True
        for _i in range(depth - 1):
            model = SMMModel(ssm_param_strategy=ssm_param_strategy,
                             ssm_calc_strategy=ssm_calc_strategy,
                             num_hidden_state=num_hidden_state,
                             input_dim=input_dim if first else num_hidden_state,
                             output_dim=num_hidden_state,
                             trainable_param_list=trainable_param_list,
                             device=device,
                             non_linearity=non_linearity)
            self.layers.append(model)
            first = False

        model = SMMModel(ssm_param_strategy=ssm_param_strategy,
                         ssm_calc_strategy=ssm_calc_strategy,
                         num_hidden_state=num_hidden_state,
                         input_dim=input_dim if first else num_hidden_state,
                         output_dim=output_dim,
                         trainable_param_list=trainable_param_list,
                         device=device,
                         non_linearity=non_linearity)
        self.layers.append(model)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        hiddens, outputs = self.layers[0].forward_with_hiddens(x)
        for layer in self.layers[1:]:
            hiddens, outputs = layer.forward_with_hiddens(hiddens)
        return outputs

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
        As = []
        Bs = []
        Cs = []
        Ds = []
        for layer in self.layers:
            A, B, C, D = layer.get_params()

            As.append(A)
            Bs.append(B)
            Cs.append(C)
            Ds.append(D)

        return As, Bs, Cs, Ds

    def get_num_hidden_state(self):
        return self.num_hidden_state

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def get_depth(self):
        return self.output_dim
