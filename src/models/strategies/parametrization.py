from src.models.strategies.base import BaseSMMParametrizationStrategy


class DiscreteSMMParametrizationStrategy(BaseSMMParametrizationStrategy):

    def __init__(self, ssm_init_strategy, ssm_storing_strategy):
        self.init_strategy = ssm_init_strategy
        self.ssm_storing_strategy = ssm_storing_strategy
        super().__init__(ssm_init_strategy, ssm_storing_strategy)

    def init_param(self, num_hidden_state, input_dim, output_dim, device):
        p_A, p_B, p_C, p_D = self.init_strategy.get_init_params(
            num_hidden_state=num_hidden_state,
            input_dim=input_dim,
            output_dim=output_dim
        )

        p_A, p_B, p_C, p_D = self.ssm_storing_strategy.store(p_A, p_B, p_C, p_D)

        return p_A, p_B, p_C, p_D

    def get_param(self, p_A, p_B, p_C, p_D, device):
        p_A, p_B, p_C, p_D = self.ssm_storing_strategy.load(p_A, p_B, p_C, p_D)
        return p_A, p_B, p_C, p_D


class ContinuousSMMParametrizationStrategy(BaseSMMParametrizationStrategy):

    def __init__(self,
                 ssm_init_strategy,
                 ssm_storing_strategy,
                 ssm_discretization_strategy,
                 dt,
                 discretize_parameters=False):

        self.init_strategy = ssm_init_strategy
        self.discretization_strategy = ssm_discretization_strategy
        self.dt = dt
        self.discretize_parameters = discretize_parameters
        self.ssm_storing_strategy = ssm_storing_strategy
        super().__init__(ssm_init_strategy, ssm_storing_strategy)

    def init_param(self, num_hidden_state, input_dim, output_dim, device):
        p_A, p_B, p_C, p_D = self.init_strategy.get_init_params(
            num_hidden_state=num_hidden_state,
            input_dim=input_dim,
            output_dim=output_dim
        )

        p_A = p_A.to(device)
        p_B = p_B.to(device)
        p_C = p_C.to(device)
        p_D = p_D.to(device)


        if self.discretize_parameters:
            p_A, p_B, p_C, p_D = self.discretization_strategy.discretize(
                self.dt, p_A, p_B, p_C, p_D, device)

        return self.ssm_storing_strategy.store(p_A, p_B, p_C, p_D)

    def get_param(self, p_A, p_B, p_C, p_D, device):
        p_A, p_B, p_C, p_D = self.ssm_storing_strategy.load(p_A, p_B, p_C, p_D)

        if not self.discretize_parameters:
            p_A, p_B, p_C, p_D = self.discretization_strategy.discretize(
                self.dt, p_A, p_B, p_C, p_D, device)

        return p_A, p_B, p_C, p_D
