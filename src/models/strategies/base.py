class BaseSMMCalcStrategy:

    def __init__(self):
        pass

    def calc(self, x, A, B, C, D, device):
        raise NotImplementedError("SMMCalcStrategy needs to implement this func")


class BaseSMMParametrizationStrategy:

    def __init__(self, ssm_init_strategy, ssm_storing_strategy):
        pass

    def init_param(self, num_hidden_state, input_dim, output_dim):
        raise NotImplementedError("SMMCalcStrategy needs to implement this func")

    def get_param(self, p_A, p_B, p_C, p_D):
        raise NotImplementedError("SMMCalcStrategy needs to implement this func")


class BaseSSMInitStrategy:

    def __init__(self):
        pass

    def get_init_params(self, num_hidden_state, input_dim, output_dim):
        raise NotImplementedError("BaseInitStrategy needs to implement this func")


class BaseSSMDiscretizationStrategy:

    def __init__(self):
        pass

    def discretize(self, dt, A, B, C, D):
        raise NotImplementedError("BaseInitStrategy needs to implement this func")


class BaseSSMStoringStrategy:

    def __init__(self):
        pass

    def store(self, A, B, C, D):
        raise NotImplementedError("BaseSSMStoringStrategy needs to implement this func")

    def load(self, A, B, C, D):
        raise NotImplementedError("BaseSSMStoringStrategy needs to implement this func")
