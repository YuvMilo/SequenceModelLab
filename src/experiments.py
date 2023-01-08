import math
import torch
from torch.utils.data import DataLoader

from src.data_loaders.data_sets.delay_data_set import DelayedSignalDataset,\
        DelayedSignalDatasetRegenerated
from src.loggers.base_logger import BaseLogger
from src.data_loaders.data_sets.utils.signals import WhiteSignalGenerator
from src.training import train
from src.loss_function.delay_loss_function import delay_l2
from src.models.ssm import SMMModel

import src.models.strategies.storing as storing_strat
import src.models.strategies.init as init_strat
import src.models.strategies.discretization as disc_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat

from src.models.LRNN import LinearRNN

def get_LRNN(num_hidden_state):
    rnn = LinearRNN(hidden_dim=num_hidden_state)
    return rnn

def get_diag_hippo_model(num_hidden_state):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.DiagHippoInitStrategy(),
            ssm_storing_strategy=storing_strat.ComplexAs2DRealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearDiagSSMDiscretizationStrategy(),
            discretize_parameters=False,
            dt=0.01
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=["C", "A"]
    )


def get_hippo_diag_model_disc_param(num_hidden_state):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.DiagHippoInitStrategy(),
            ssm_storing_strategy=storing_strat.ComplexAs2DRealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearDiagSSMDiscretizationStrategy(),
            discretize_parameters=True,
            dt=0.01
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=["A", "C"]
    )


def get_hippo_diag_model_with_low_imag(num_hidden_state):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleHippoInitStrategy(
                A_init_func=lambda n: 0.5 * torch.ones(n) + \
                                      1j * math.pi * torch.arange(n) * (10/n),
                B_init_func=lambda n: torch.ones([n, 1], dtype=torch.cfloat),
                C_init_func=lambda n: torch.randn([1, n], dtype=torch.cfloat),
                D_init_func=lambda n: torch.zeros([n, 1], dtype=torch.cfloat),

            ),
            ssm_storing_strategy=storing_strat.ComplexAs2DRealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearDiagSSMDiscretizationStrategy(),
            discretize_parameters=False,
            dt=0.01
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=["A", "C"]
    )


def get_full_ssm(num_hidden_state,
                 A_init_func,
                 B_init_std=1e-1,
                 C_init_std=1e-1
                 ):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleInitStrategy(
                A_init_func=A_init_func,
                B_init_func=lambda n: B_init_std * torch.randn([n, 1], dtype=torch.float),
                C_init_func=lambda n: C_init_std * torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([n, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"]
    )

def get_full_delay_ssm(num_hidden_state,
                 delay):
    def get_delay_A(n):
        A = torch.zeros([n,n])
        for i in range(delay):
            A[i+1, i] = 1
        return A

    def get_delay_B(n):
        B = torch.zeros([n, 1])
        B[0, 0] = 1
        return B

    def get_delay_C(n):
        C = torch.zeros([1, n])
        C[0, delay] = 1
        C = C - 0.1
        return C

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleInitStrategy(
                A_init_func=get_delay_A,
                B_init_func=lambda n: torch.randn([n, 1], dtype=torch.float)/5,
                C_init_func=lambda n: torch.randn([1, n], dtype=torch.float)/5,
                D_init_func=lambda n: torch.zeros([n, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["C"]
    )

def get_cont_full_ssm(num_hidden_state,
                      A_init_std=1e-4,
                      B_init_std=1e-1,
                      C_init_std=1e-1
                 ):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleInitStrategy(
                A_init_func=lambda n: -0.3 * torch.eye(n) + A_init_std*torch.randn([n, n], dtype=torch.float),
                B_init_func=lambda n: B_init_std*torch.randn([n, 1], dtype=torch.float),
                C_init_func=lambda n: C_init_std*torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([n, 1], dtype=torch.float),

            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearSSMDiscretizationStrategy(),
            discretize_parameters=False,
            dt=0.01
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"]
    )


def main():
    lag = 20
    seq_len = 100+lag
    samples_num = 100
    hidden_size = 400

    criterion = delay_l2(lag)
    signal_generator = WhiteSignalGenerator(freq=1, dt=0.05)

    ds = DelayedSignalDataset(lag_length=lag, samples_num=samples_num,
                              seq_length=seq_len, signal_generator=signal_generator)
    ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
                              seq_length=seq_len, signal_generator=signal_generator)
    dl = DataLoader(ds, batch_size=samples_num)

    logger = BaseLogger()

    hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)
    hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)
    hippo_diag_model_with_low_imag = get_hippo_diag_model_with_low_imag(num_hidden_state=hidden_size)

    full_ssm = get_full_ssm(num_hidden_state=hidden_size,
                            A_init_func=lambda n: 0.01*torch.eye(n) + torch.randn([n, n]) * 0.0001,
                            B_init_std=1/(hidden_size**0.5),
                            C_init_std=1/(hidden_size**0.5))
    cont_full_ssm = get_cont_full_ssm(num_hidden_state=hidden_size)
    delay_ssm = get_full_delay_ssm(num_hidden_state=hidden_size,
                                   delay=lag)
    # lrnn = get_LRNN(num_hidden_state=hidden_size)

    model = full_ssm
    # model = hippo_diag_model
    # model = hippo_diag_model_with_low_imag
    # model = hippo_diag_model_disc_param

    #voptimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    optimizer = torch.optim.Adam(model.parameters())
    train(model=model,
          dl=dl,
          logger=logger,
          criterion=criterion,
          num_epochs=10000,
          optimizer=optimizer)

    pass


if __name__ == "__main__":
    main()
