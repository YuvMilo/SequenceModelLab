import math
import os
import torch
from torch.utils.data import DataLoader

from src.data_loaders.data_sets.delay_data_set import DelayedSignalDatasetRegenerated
from src.loggers.ssm_logger import SSMLogger
import src.data_loaders.data_sets.utils.signals as signals
from src.training import train, train_smm_random_noise_fast
from src.loss_function.delay_loss_function import delay_l2
from src.models.ssm import SMMModel

import src.models.strategies.storing as storing_strat
import src.models.strategies.init as init_strat
import src.models.strategies.discretization as disc_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat


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
                A_init_func=lambda n: 0.5 * torch.ones(n) +
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


def get_full_delay_ssm(num_hidden_state, delay):
    def get_delay_A(n):
        A = torch.zeros([n, n])
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
        trainable_param_list=["A", "C"]
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
                A_init_func=lambda n: -0.5 * torch.eye(n) + A_init_std*torch.randn([n, n], dtype=torch.float),
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


def experiment_with_lag():
    start_lag = 10
    end_lag = 50
    lag_jumps = 5
    seq_len = 400
    hidden_size = 200
    epochs = 10000
    samples_num = 1000

    def get_saving_path_for_exp(lag, model):
        return ".\\results\\lag_exp\\"+model+"_"+str(lag)

    signal_generator = signals.NormalNoiseSignalGenerator()  # signals.NormalNoiseCumSignalGenerator()

    for lag in range(start_lag, end_lag, lag_jumps):
        print("current_lag=", lag)
        criterion = delay_l2(lag)

        ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
                                             seq_length=seq_len, signal_generator=signal_generator)
        dl = DataLoader(ds, batch_size=samples_num)

        logger_hippo = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="hippo"))
        hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)

        logger_disc_hippo = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="hippo_disc"))
        hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)

        logger_fssm = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="fssm"))
        full_ssm = get_full_ssm(num_hidden_state=hidden_size,
                                A_init_func=lambda n: 0.1 * torch.eye(n) + torch.randn([n, n]) * 0.0001,
                                B_init_std=1 / (hidden_size ** 0.5),
                                C_init_std=1 / (hidden_size ** 0.5))

        optimizer = torch.optim.Adam(full_ssm.parameters())
        train(model=full_ssm,
              dl=dl,
              logger=logger_fssm,
              criterion=criterion,
              num_epochs=epochs,
              optimizer=optimizer)

        optimizer = torch.optim.Adam(hippo_diag_model.parameters())
        train(model=hippo_diag_model,
              dl=dl,
              logger=logger_hippo,
              criterion=criterion,
              num_epochs=epochs,
              optimizer=optimizer)

        optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
        train(model=hippo_diag_model_disc_param,
              dl=dl,
              logger=logger_disc_hippo,
              criterion=criterion,
              num_epochs=epochs,
              optimizer=optimizer)


def experiment_with_lag_fast_loss():
    start_lag = 3
    end_lag = 80
    lag_jumps = 3
    seq_len = 400
    hidden_size = 133
    epochs = 5000

    def get_saving_path_for_exp(lag, model):
        return ".\\results\\lag_exp\\"+"fast_"+model+"_"+str(lag)

    for lag in range(start_lag, end_lag, lag_jumps):
        hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)
        hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)
        full_ssm_big_diag = get_full_ssm(
            num_hidden_state=hidden_size,
            A_init_func=lambda n: 0.9 * torch.eye(n) + 0.01 * torch.randn([n, n])/n,
            B_init_std=1 / (hidden_size ** 0.5),
            C_init_std=1 / (hidden_size ** 0.5)
        )
        full_ssm_small_diag = get_full_ssm(
            num_hidden_state=hidden_size,
            A_init_func=lambda n: 0.8 * torch.eye(n) + 0.01 * torch.randn([n, n])/n,
            B_init_std=1 / (hidden_size ** 0.5),
            C_init_std=1 / (hidden_size ** 0.5)
        )

        saving_path = get_saving_path_for_exp(lag, model="fssm_bd")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_ssm_big_diag.parameters())
            train_smm_random_noise_fast(
                model=full_ssm_big_diag,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                min_cut=epochs
            )

        saving_path = get_saving_path_for_exp(lag, model="fssm_sd")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_ssm_small_diag.parameters())
            train_smm_random_noise_fast(
                model=full_ssm_small_diag,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                min_cut=epochs
            )

        saving_path = get_saving_path_for_exp(lag, model="hippo")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(hippo_diag_model.parameters())
            train_smm_random_noise_fast(
                  model=hippo_diag_model,
                  lag=lag,
                  seq_len=seq_len,
                  logger=logger,
                  num_epochs=epochs,
                  optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="hippo_disc")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
            train_smm_random_noise_fast(
                  model=hippo_diag_model_disc_param,
                  lag=lag,
                  seq_len=seq_len,
                  logger=logger,
                  num_epochs=epochs,
                  optimizer=optimizer)


def playing():
    lag = 100
    seq_len = lag*3
    samples_num = 100
    hidden_size = 50

    criterion = delay_l2(lag)
    # signal_generator = signals.WhiteSignalGenerator(freq=1, dt=0.1)
    # signal_generator = signals.NormalNoiseSignalGenerator()
    signal_generator = signals.NormalNoiseCumSignalGenerator()

    # ds = DelayedSignalDataset(lag_length=lag, samples_num=samples_num,
    #                           seq_length=seq_len, signal_generator=signal_generator)
    ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
                                         seq_length=seq_len, signal_generator=signal_generator)
    dl = DataLoader(ds, batch_size=samples_num)

    logger = SSMLogger("playing_log")

    hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)
    hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)
    hippo_diag_model_with_low_imag = get_hippo_diag_model_with_low_imag(num_hidden_state=hidden_size)

    # full_ssm = get_full_ssm(num_hidden_state=hidden_size,
    #                         A_init_func=lambda n: 0.8*torch.eye(n) + torch.randn([n, n]) * 0.0001,
    #                         B_init_std=1/(hidden_size**0.5),
    #                         C_init_std=1/(hidden_size**0.5))
    # cont_full_ssm = get_cont_full_ssm(num_hidden_state=hidden_size)
    # delay_ssm = get_full_delay_ssm(num_hidden_state=hidden_size,
    #                                delay=lag)
    # lrnn = get_LRNN(num_hidden_state=hidden_size)

    # model = full_ssm
    # model = hippo_diag_model
    # model = hippo_diag_model_with_low_imag
    model = hippo_diag_model

    # voptimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    optimizer = torch.optim.Adam(model.parameters())
    train(model=model,
          dl=dl,
          logger=logger,
          criterion=criterion,
          num_epochs=10000,
          optimizer=optimizer)

    pass


if __name__ == "__main__":
    # experiment_with_lag()
    experiment_with_lag_fast_loss()
