import os
import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data_loaders.data_sets.delay_data_set import DelayedSignalDatasetRegenerated
from src.loggers.ssm_logger import SSMLogger
import src.data_loaders.data_sets.utils.signals as signals
from src.training import train, train_smm_random_noise_fast
from src.loss_function.delay_loss_function import delay_l2
from src.models.ssm import SMMModel

import src.models.strategies.storing as storing_strat
import src.models.strategies.ssm_init as init_strat
import src.models.strategies.discretization as disc_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_diag_hippo_model(num_hidden_state, device):
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
        trainable_param_list=["C", "A"],
        device=device
    )

def get_diag_real_model(num_hidden_state,
                        device,
                        A_init_func,
                        C_init_std=0.1,
                        ):
        return SMMModel(
            num_hidden_state=num_hidden_state,
            input_dim=1,
            output_dim=1,
            ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
                ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                    A_init_func=A_init_func,
                    B_init_func=lambda n: torch.ones([n, 1], dtype=torch.float),
                    C_init_func=lambda n: C_init_std * torch.randn([1, n], dtype=torch.float),
                    D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
                ),
                ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
            ),
            ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
            trainable_param_list=["A", "C"],
            device=device
        )

def get_diag_hippo_model(num_hidden_state, device):
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
        trainable_param_list=["C", "A"],
        device=device
    )

def get_hippo_diag_model_disc_param(num_hidden_state, device):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.DiagHippoInitStrategy(),
            ssm_storing_strategy=storing_strat.ComplexAs2DRealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearDiagSSMDiscretizationStrategy(),
            discretize_parameters=True,
            dt=0.05
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=["A", "C"],
        device=device
    )


def get_hippo_diag_model_with_low_imag(num_hidden_state, device):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleDiagHippoInitStrategy(
                A_init_func=lambda n: 0.5 * torch.ones(n) +
                                      1j * math.pi * torch.arange(n) * (10/n),
                B_init_func=lambda n: torch.ones([n, 1], dtype=torch.cfloat),
                C_init_func=lambda n: torch.randn([1, n], dtype=torch.cfloat),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.cfloat),

            ),
            ssm_storing_strategy=storing_strat.ComplexAs2DRealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearDiagSSMDiscretizationStrategy(),
            discretize_parameters=False,
            dt=0.05
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=["A", "C"],
        device=device
    )


def get_full_disc_hippo_model(num_hidden_state,
                              device,
                              dt=0.01,
                              C_init_std=1,
                              disc_only_onces = False):

    def A_hippo_innit_func(N):
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)

        # freqs = np.arange(N // 2)
        # d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        # A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        # B = np.zeros(N)
        # B[0::2] = 2 ** .5
        # B[0] = 1
        # A = A - B[:, None] * B[None, :]
        #
        A = torch.Tensor(A)
        return A

    def B_hippo_init_func(N):
        # freqs = np.arange(N // 2)
        # d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        # A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        # B = np.zeros(N)
        # B[0::2] = 2 ** .5
        # B[0] = 1
        # B = B[:, None]

        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()

        B = torch.Tensor(B)
        return B

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=A_hippo_innit_func,
                B_init_func=B_hippo_init_func,
                C_init_func=lambda n: C_init_std * torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
            dt=dt,
            ssm_discretization_strategy=disc_strat.BilinearSSMDiscretizationStrategy(),
            discretize_parameters=disc_only_onces
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"],
        device=device
    )

def get_full_lagT_model(num_hidden_state,
                              device,
                              dt=0.01,
                              C_init_std=1,
                              disc_only_onces = False):

    def A_lagT_innit_func(N):
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        A = torch.Tensor(A)
        return A

    def B_lagT_init_func(N):
        B = np.ones((N, 1))*0.5
        B = torch.Tensor(B)
        return B

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=A_lagT_innit_func,
                B_init_func=B_lagT_init_func,
                C_init_func=lambda n: C_init_std * torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
            dt=dt,
            ssm_discretization_strategy=disc_strat.BilinearSSMDiscretizationStrategy(),
            discretize_parameters=disc_only_onces
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"],
        device=device
    )


def get_full_ssm(num_hidden_state,
                 device,
                 A_init_func,
                 B_init_std=1e-1,
                 C_init_std=1e-1,
                 B_func=None
                 ):

    if B_func is None:
        B_func = lambda n: B_init_std * torch.randn([n, 1], dtype=torch.float),

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=A_init_func,
                B_init_func=B_func,
                C_init_func=lambda n: C_init_std * torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"],
        device=device
    )


def get_full_delay_ssm(num_hidden_state, delay, device):
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
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=get_delay_A,
                B_init_func=lambda n: torch.randn([n, 1], dtype=torch.float)/5,
                C_init_func=lambda n: torch.randn([1, n], dtype=torch.float)/5,
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "C"],
        device=device
    )


def get_const_kernel_smm(num_hidden_state, decay, kernel_val, device):
    def get_delay_A(n):
        A = torch.zeros([n, n])
        for i in range(n-1):
            A[i+1, i] = 1
        return A*decay

    def get_delay_B(n):
        B = torch.zeros([n, 1])
        B[0, 0] = 1
        return B

    def get_delay_C(n):
        C = torch.ones([1, n])
        return C*kernel_val

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=get_delay_A,
                B_init_func=get_delay_B,
                C_init_func=get_delay_C,
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "C", "B"],
        device=device
    )


def get_cont_full_ssm(num_hidden_state,
                      device,
                      A_init_std=1e-4,
                      B_init_std=1e-1,
                      C_init_std=1e-1,
                      ):
    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.ContinuousSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=lambda n: -0.5 * torch.eye(n) + A_init_std*torch.randn([n, n], dtype=torch.float),
                B_init_func=lambda n: B_init_std*torch.randn([n, 1], dtype=torch.float),
                C_init_func=lambda n: C_init_std*torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),

            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
            ssm_discretization_strategy=disc_strat.BilinearSSMDiscretizationStrategy(),
            discretize_parameters=False,
            dt=0.05
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"],
        device=device
    )


# def experiment_with_lag():
#     start_lag = 10
#     end_lag = 50
#     lag_jumps = 5
#     seq_len = 400
#     hidden_size = 200
#     epochs = 10000
#     samples_num = 1000
#
#     def get_saving_path_for_exp(lag, model):
#         return ".\\results\\lag_exp\\"+model+"_"+str(lag)
#
#     signal_generator = signals.NormalNoiseSignalGenerator()  # signals.NormalNoiseCumSignalGenerator()
#
#     for lag in range(start_lag, end_lag, lag_jumps):
#         print("current_lag=", lag)
#         criterion = delay_l2(lag)
#
#         ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
#                                              seq_length=seq_len, signal_generator=signal_generator)
#         dl = DataLoader(ds, batch_size=samples_num)
#
#         logger_hippo = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="hippo"))
#         hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size,
#                                                 device=device)
#
#         logger_disc_hippo = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="hippo_disc"))
#         hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size,
#                                                                       device=device)
#
#         logger_fssm = SSMLogger(saving_path=get_saving_path_for_exp(lag, model="fssm"))
#         full_ssm = get_full_ssm(num_hidden_state=hidden_size,
#                                 device=device,
#                                 A_init_func=lambda n: 0.1 * torch.eye(n) + torch.randn([n, n]) * 0.0001,
#                                 B_init_std=1 / (hidden_size ** 0.5),
#                                 C_init_std=1 / (hidden_size ** 0.5))
#
#         optimizer = torch.optim.Adam(full_ssm.parameters())
#         train(model=full_ssm,
#               dl=dl,
#               logger=logger_fssm,
#               criterion=criterion,
#               num_epochs=epochs,
#               optimizer=optimizer)
#
#         optimizer = torch.optim.Adam(hippo_diag_model.parameters())
#         train(model=hippo_diag_model,
#               dl=dl,
#               logger=logger_hippo,
#               criterion=criterion,
#               num_epochs=epochs,
#               optimizer=optimizer)
#
#         optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
#         train(model=hippo_diag_model_disc_param,
#               dl=dl,
#               logger=logger_disc_hippo,
#               criterion=criterion,
#               num_epochs=epochs,
#               optimizer=optimizer)


def playing():
    lag = 128
    seq_len = 512
    hidden_size = 64
    epochs = 2000

    # criterion = delay_l2(lag)
    # signal_generator = signals.WhiteSignalGenerator(freq=1, dt=0.01)
    # # signal_generator = signals.NormalNoiseSignalGenerator()
    # # signal_generator = signals.NormalNoiseCumSignalGenerator()
    #
    # # ds = DelayedSignalDataset(lag_length=lag, samples_num=samples_num,
    # #                           seq_length=seq_len, signal_generator=signal_generator)
    # ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
    #                                      seq_length=seq_len, signal_generator=signal_generator)
    # dl = DataLoader(ds, batch_size=samples_num)

    logger = SSMLogger(os.path.join("..", "results", "playing_res"))

    # hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)
    # hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)
    # hippo_diag_model_with_low_imag = get_hippo_diag_model_with_low_imag(num_hidden_state=hidden_size)

    full_hippo_model_disc_bdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                                          device=device,
                                                          dt=0.001,
                                                          disc_only_onces=False)
    # full_hippo_model_disc_sdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
    #                                                       device=device,
    #                                                       dt=0.001,
    #                                                       disc_only_onces=True)

    def get_diag_i(i,n):
        return 1*((n-(i+1))**2/(n**2)) + 0.5*(1-(n-(i+1))**2/(n**2))

    def get_under_diag_i(i,n):
        return -1 * (1-get_diag_i(i, n)+(i**2)/(n**3))

    def get_A(n):
        A = torch.zeros([n,n])
        for i in range(n):
            A[i,i] = get_diag_i(i,n)
        for i in range(n-1):
            A[i+1,i] = get_under_diag_i(i,n)
        import matplotlib.pyplot as plt;
        plt.imshow(A, cmap='hot', interpolation='nearest');
        plt.colorbar();
        plt.show()
        return A

    spread = 15
    def get_B(n):
        B = torch.zeros([n,1])
        B[:spread, 0] = 1/spread
        return B

    full_ssm_big_diag = get_full_ssm(
        num_hidden_state=hidden_size,
        device=device,
        A_init_func=get_A,
        B_func=get_B,
        B_init_std=1 / (hidden_size ** 0.5),
        C_init_std=1 / (hidden_size ** 0.5)
    )



    # full_hipo_model = get_diag_hippo_model(num_hidden_state=hidden_size, device=device)

    # diag_real_model = get_diag_real_model(
    #     num_hidden_state=hidden_size,
    #     device=device,
    #     A_init_func=lambda n: torch.linspace(-0.9, 0.9, n)
    # )

    # const_kernel_model = get_const_kernel_smm(
    #     num_hidden_state=hidden_size,
    #     device=device,
    #     decay=0.99,
    #     kernel_val=1/hidden_size,
    # )

    #full_hipo_model = get_full_ssm(num_hidden_state=hidden_size,
    #                        A_init_func=lambda n: 0.8*torch.eye(n) + torch.randn([n, n]) * 0.0001,
    #                        B_init_std=1/(hidden_size**0.5),
    #                        C_init_std=1/(hidden_size**0.5),
    #                        device=device)
    # cont_full_ssm = get_cont_full_ssm(num_hidden_state=hidden_size)
    # delay_ssm = get_full_delay_ssm(num_hidden_state=hidden_size,
    #                                delay=lag)
    # lrnn = get_LRNN(num_hidden_state=hidden_size)

    # model = full_ssm
    # model = hippo_diag_model
    # model = hippo_diag_model_with_low_imag
    model = full_ssm_big_diag
    model = full_hippo_model_disc_bdt

    # voptimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    # train(model=model,
    #       dl=dl,
    #       logger=logger,
    #       criterion=criterion,
    #       num_epochs=10000,
    #       optimizer=optimizer)

    train_smm_random_noise_fast(
        model=model,
        lag=lag,
        seq_len=seq_len,
        logger=logger,
        num_epochs=epochs,
        optimizer=optimizer,
        min_cut=10000,
        plot=True
    )

    pass


def exp_fssm():
    start_lag = 10
    end_lag = 512
    lag_jumps = 10
    seq_len = 2048
    hidden_size = 128
    epochs = 2000

    def get_saving_path_for_exp(lag, model):
        return os.path.join("..", "results", "lag_exp", "fast_" + model + "_" + str(lag))
        #return ".\\results\\lag_exp\\"+"fast_"+model+"_"+str(lag)

    for lag in range(start_lag, end_lag, lag_jumps):
        #hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size, device=device)
        #hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size, device=device)
        full_ssm_big_diag = get_full_ssm(
            num_hidden_state=hidden_size,
            device=device,
            A_init_func=lambda n: 0.9 * torch.eye(n) + 0.01 * torch.randn([n, n])/n,
            B_init_std=1 / (hidden_size ** 0.5),
            C_init_std=1 / (hidden_size ** 0.5)
        )
        full_ssm_small_diag = get_full_ssm(
            device=device,
            num_hidden_state=hidden_size,
            A_init_func=lambda n: 0.8 * torch.eye(n) + 0.01 * torch.randn([n, n])/n,
            B_init_std=1 / (hidden_size ** 0.5),
            C_init_std=1 / (hidden_size ** 0.5)
        )

        full_ssm_huge_diag = get_full_ssm(
            num_hidden_state=hidden_size,
            device=device,
            A_init_func=lambda n: 0.95 * torch.eye(n) + 0.01 * torch.randn([n, n]) / n,
            B_init_std=1 / (hidden_size ** 0.5),
            C_init_std=1 / (hidden_size ** 0.5)
        )

        # saving_path = get_saving_path_for_exp(lag, model="fssm_sd")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path) as logger:
        #     optimizer = torch.optim.Adam(full_ssm_small_diag.parameters())
        #     train_smm_random_noise_fast(
        #         model=full_ssm_small_diag,
        #         lag=lag,
        #         seq_len=seq_len,
        #         logger=logger,
        #         num_epochs=epochs,
        #         optimizer=optimizer,
        #         min_cut=epochs
        #     )

        saving_path = get_saving_path_for_exp(lag, model="fssm_bd")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_ssm_big_diag.parameters())
            train_smm_random_noise_fast(
                model=full_ssm_big_diag,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                min_cut=epochs//2
            )

        saving_path = get_saving_path_for_exp(lag, model="fssm_ud")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_ssm_huge_diag.parameters())
            train_smm_random_noise_fast(
                model=full_ssm_huge_diag,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                min_cut=epochs//2
            )

        # saving_path = get_saving_path_for_exp(lag, model="hippo")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)
        #
        # saving_path = get_saving_path_for_exp(lag, model="hippo_disc")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model_disc_param,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)


HIPPO_DISC_ONCES_EXP_LR = 0.0001
def exp_hippo():
    start_lag = 10
    end_lag = 512
    lag_jumps = 10
    seq_len = 2048
    hidden_size = 128
    epochs = 2000

    def get_saving_path_for_exp(lag, model):
        return os.path.join("..", "results", "lag_exp_long", "fast_" + model + "_" + str(lag))
        #return ".\\results\\lag_exp_long\\"+"fast_"+model+"_"+str(lag)

    for lag in range(start_lag, end_lag, lag_jumps):
        # hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size, device=device)
        # hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size, device=device)
        # full_hippo_model_bdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
        #                                                      device=device,
        #                                                      dt=0.1)
        # full_hippo_model_sdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
        #                                                 device=device,
        #                                                 dt=0.001)

        full_hippo_model_disc_bdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                                         device=device,
                                                         dt=0.1,
                                                         disc_only_onces=True)
        full_hippo_model_disc_sdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                                         device=device,
                                                         dt=0.001,
                                                         disc_only_onces=True)

        # saving_path = get_saving_path_for_exp(lag, model="diag_hippo")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)
        #
        # saving_path = get_saving_path_for_exp(lag, model="diag_hippo_disc")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model_disc_param,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)
        #
        # saving_path = get_saving_path_for_exp(lag, model="full_hippo_bdt")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(full_hippo_model_bdt.parameters())
        #     train_smm_random_noise_fast(
        #         model=full_hippo_model_bdt,
        #         lag=lag,
        #         seq_len=seq_len,
        #         logger=logger,
        #         num_epochs=epochs,
        #         optimizer=optimizer)
        #
        # saving_path = get_saving_path_for_exp(lag, model="full_hippo_sdt")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(full_hippo_model_sdt.parameters())
        #     train_smm_random_noise_fast(
        #         model=full_hippo_model_sdt,
        #         lag=lag,
        #         seq_len=seq_len,
        #         logger=logger,
        #         num_epochs=epochs,
        #         optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_hippo_disc_bdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_hippo_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_random_noise_fast(
                model=full_hippo_model_disc_bdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_hippo_disc_sdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_hippo_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_random_noise_fast(
                model=full_hippo_model_disc_sdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)


def exp_lagT():
    start_lag = 10
    end_lag = 512
    lag_jumps = 10
    seq_len = 2048
    hidden_size = 128
    epochs = 2000

    def get_saving_path_for_exp(lag, model):
        return os.path.join("..", "results", "lag_exp_long", "fast_" + model + "_" + str(lag))
        #return ".\\results\\lag_exp_long\\"+"fast_"+model+"_"+str(lag)

    for lag in range(start_lag, end_lag, lag_jumps):
        # hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size, device=device)
        # hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size, device=device)
        full_lagT_model_bdt = get_full_lagT_model(num_hidden_state=hidden_size,
                                                             device=device,
                                                             dt=0.1)
        full_lagT_model_sdt = get_full_lagT_model(num_hidden_state=hidden_size,
                                                        device=device,
                                                        dt=0.001)

        full_lagT_model_disc_bdt = get_full_lagT_model(num_hidden_state=hidden_size,
                                                         device=device,
                                                         dt=0.1,
                                                         disc_only_onces=True)
        full_lagT_model_disc_sdt = get_full_lagT_model(num_hidden_state=hidden_size,
                                                         device=device,
                                                         dt=0.001,
                                                         disc_only_onces=True)

        # saving_path = get_saving_path_for_exp(lag, model="diag_hippo")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)
        #
        # saving_path = get_saving_path_for_exp(lag, model="diag_hippo_disc")
        # print(os.path.basename(saving_path))
        # with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
        #     optimizer = torch.optim.Adam(hippo_diag_model_disc_param.parameters())
        #     train_smm_random_noise_fast(
        #           model=hippo_diag_model_disc_param,
        #           lag=lag,
        #           seq_len=seq_len,
        #           logger=logger,
        #           num_epochs=epochs,
        #           optimizer=optimizer)
        #
        saving_path = get_saving_path_for_exp(lag, model="full_lagT_bdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_lagT_model_bdt.parameters())
            train_smm_random_noise_fast(
                model=full_lagT_model_bdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_lagT_sdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_lagT_model_sdt.parameters())
            train_smm_random_noise_fast(
                model=full_lagT_model_sdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_lagT_disc_bdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_lagT_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_random_noise_fast(
                model=full_lagT_model_disc_bdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_lagT_disc_sdt")
        print(os.path.basename(saving_path))
        with SSMLogger(saving_path=saving_path, kernel_saving_size=seq_len) as logger:
            optimizer = torch.optim.Adam(full_lagT_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_random_noise_fast(
                model=full_lagT_model_disc_sdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

if __name__ == "__main__":
    # experiment_with_lag()
    # makeing_missed()
    # experiment_with_lag_fast_loss()
    # experiment_with_big_seq()
    playing()

