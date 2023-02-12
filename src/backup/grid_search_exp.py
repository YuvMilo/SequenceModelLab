import math
import os
import torch
import numpy as np
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.training.train_ssm import train_smm_over_white_noise_lag_multiprocess
from src.models.ssm import SMMModel

import src.models.strategies.storing as storing_strat
import src.models.strategies.ssm_init as init_strat
import src.models.strategies.discretization as disc_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat

from itertools import product

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
                              disc_only_onces = False,
                              trainable_param_list = ["A", "B", "C"]):

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
        trainable_param_list=trainable_param_list,
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
        B_func = lambda n: B_init_std * torch.randn([n, 1], dtype=torch.float)

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


def polar_to_complex(radii, angles_radians):
    return radii * np.exp(1j*angles_radians)


def get_rot_ssm_equally_spaced(num_hidden_state, device):
    #
    # eps = 0.005

    # def get_real_i(i):
    #     return i / num_hidden_state * (1 - eps)
    #
    # def get_imag_i(i):
    #     return np.sqrt((1 - eps) ** 2 - get_real_i(i) ** 2)

    radii = 0.99
    eff_hidden = num_hidden_state//2

    # def get_i_eigen(i):
    #     return polar_to_complex(radii, i * np.pi / (eff_hidden - 1))

    def get_i_eigen(i):
        # return polar_to_complex(radii, np.pi/2*(1-(1/(i+1))**0.5))
        angle = np.pi * (i / (eff_hidden - 1))
        angle += 2 ** 0.5
        angle = angle % np.pi
        return polar_to_complex(radii, angle)

    def get_real_i(i):
        return get_i_eigen(i).real

    def get_imag_i(i):
        return get_i_eigen(i).imag

    def get_rot_i(i):
        rot = torch.zeros([2, 2])
        # this matrix has eigenvalue get_real_i(i) +- i*get_imag_i(i)
        rot[0, 0] = get_real_i(i)
        rot[1, 1] = get_real_i(i)
        rot[0, 1] = -1 * get_imag_i(i)
        rot[1, 0] = get_imag_i(i)
        return rot

    def get_A(n):
        A = torch.zeros([n, n])
        for i in range(0, n, 2):
            A[i:i + 2, i:i + 2] = get_rot_i(i // 2)
        return A

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=get_A,
                B_init_func=lambda n: 0.2*torch.randn([n, 1], dtype=torch.float),
                C_init_func=lambda n: 0.2*torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["C", "B", "A"],
        device=device
    )

def get_rot_ssm_one_over_n(num_hidden_state, device):

    eps = 0.005

    # def get_real_i(i):
    #     return i / num_hidden_state * (1 - eps)
    #
    # def get_imag_i(i):
    #     return np.sqrt((1 - eps) ** 2 - get_real_i(i) ** 2)

    radii = 0.99
    eff_hidden = num_hidden_state//2

    def get_i_eigen(i):
        return polar_to_complex(radii, 2 * np.pi / (i + 1))

    def get_real_i(i):
        return get_i_eigen(i).real

    def get_imag_i(i):
        return get_i_eigen(i).imag

    def get_rot_i(i):
        rot = torch.zeros([2, 2])
        # this matrix has eigenvalue get_real_i(i) +- i*get_imag_i(i)
        rot[0, 0] = get_real_i(i)
        rot[1, 1] = get_real_i(i)
        rot[0, 1] = -1 * get_imag_i(i)
        rot[1, 0] = get_imag_i(i)
        return rot

    def get_A(n):
        A = torch.zeros([n, n])
        for i in range(0, n, 2):
            A[i:i + 2, i:i + 2] = get_rot_i(i // 2)
        return A

    return SMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=get_A,
                B_init_func=lambda n: 0.1*torch.randn([n, 1], dtype=torch.float),
                C_init_func=lambda n: 0.1*torch.randn([1, n], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A","C", "B"],
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
#         logger_hippo = SSMTrainingLogger(saving_path=get_saving_path_for_exp(lag, model="hippo"))
#         hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size,
#                                                 device=device)
#
#         logger_disc_hippo = SSMTrainingLogger(saving_path=get_saving_path_for_exp(lag, model="hippo_disc"))
#         hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size,
#                                                                       device=device)
#
#         logger_fssm = SSMTrainingLogger(saving_path=get_saving_path_for_exp(lag, model="fssm"))
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
    lag = 200
    seq_len = 2048
    hidden_size = 128
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

    logger = SSMTrainingLogger(os.path.join("../..", "results", "playing_res"),
                               saving_freq=100)

    # hippo_diag_model = get_diag_hippo_model(num_hidden_state=hidden_size)
    # hippo_diag_model_disc_param = get_hippo_diag_model_disc_param(num_hidden_state=hidden_size)
    # hippo_diag_model_with_low_imag = get_hippo_diag_model_with_low_imag(num_hidden_state=hidden_size)

    rot_model = get_rot_ssm_equally_spaced(num_hidden_state=hidden_size,
                                           device=device)

    # full_hippo_model_disc_sdt = get_full_disc_hippo_model(num_hidden_state=hidden_size,
    #                                                       device=device,
    #                                                       dt=0.001,
    #                                                       disc_only_onces=True)

    # def get_diag_i(i,n):
    #     return 1*((n-(i+1))**2/(n**2)) + 0.5*(1-(n-(i+1))**2/(n**2))
    #
    # def get_under_diag_i(i,n):
    #     return -1 * (1-get_diag_i(i, n)+(i**2)/(n**3))
    #
    # def get_A(n):
    #     A = torch.zeros([n,n])
    #     for i in range(n):
    #         A[i,i] = get_diag_i(i,n)
    #     for i in range(n-1):
    #         A[i+1,i] = get_under_diag_i(i,n)
    #     import matplotlib.pyplot as plt;
    #     plt.imshow(A, cmap='hot', interpolation='nearest');
    #     plt.colorbar();
    #     plt.show()
    #     return A
    #
    # spread = 15
    # def get_B(n):
    #     B = torch.zeros([n,1])
    #     B[:spread, 0] = 1/spread
    #     return B
    #
    # full_ssm_big_diag = get_full_ssm(
    #     num_hidden_state=hidden_size,
    #     device=device,
    #     A_init_func=get_A,
    #     B_func=get_B,
    #     B_init_std=1 / (hidden_size ** 0.5),
    #     C_init_std=1 / (hidden_size ** 0.5)
    # )



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
    model = rot_model

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters())
    # train(model=model,
    #       dl=dl,
    #       logger=logger,
    #       criterion=criterion,
    #       num_epochs=10000,
    #       optimizer=optimizer)

    train_smm_over_white_noise_lag_multiprocess(
        model=model,
        lag=lag,
        seq_len=seq_len,
        logger=logger,
        num_epochs=epochs,
        optimizer=optimizer,
        min_cut=10000,
        plot=True,
        early_stop=False
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
        return os.path.join("../..", "results", "lag_exp", "fast_" + model + "_" + str(lag))
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_ssm_big_diag.parameters())
            train_smm_over_white_noise_lag_multiprocess(
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
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_ssm_huge_diag.parameters())
            train_smm_over_white_noise_lag_multiprocess(
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        return os.path.join("../..", "results", "lag_exp_long", "fast_" + model + "_" + str(lag))
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        # with SSMTrainingLogger(saving_path=saving_path) as logger:
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
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_hippo_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_over_white_noise_lag_multiprocess(
                model=full_hippo_model_disc_bdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)

        saving_path = get_saving_path_for_exp(lag, model="full_hippo_disc_sdt")
        print(os.path.basename(saving_path))
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = torch.optim.Adam(full_hippo_model_disc_bdt.parameters(),
                                         lr=HIPPO_DISC_ONCES_EXP_LR)
            train_smm_over_white_noise_lag_multiprocess(
                model=full_hippo_model_disc_sdt,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer)


hidden_size = 64
lags = [60]
seq_len = hidden_size*2
epochs = 4000

optimizers = [torch.optim.SGD, torch.optim.Adam]
lrs_sgd = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
lrs_adam = [0.1, 0.01, 0.001, 0.0001]


def grid_search_fssm_diag():
    def get_saving_path_for_exp(optimizer, lag, lr, noise, diags_init, BC_std):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplemented()

        return os.path.join("../..", "results", "grid",
                            "ssm_h{hidden}_l{lag}_lr{lr}_n{noise}_d{diag_init}_b{BC_std}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                noise=noise,
                                diag_init=diags_init,
                                BC_std=BC_std,
                                opt=opt,
                                hidden=hidden_size
                            ))

    diags_inits = [0.9, 0.93, 0.95, 0.97]
    noises = [0.0001, 0.001, 0.01, 0.1]
    BC_stds = [1, 0.1, 0.01, 0.001]

    prod1 = product([torch.optim.SGD], lags, lrs_sgd, noises, diags_inits, BC_stds)
    prod2 = product([torch.optim.Adam], lags, lrs_adam, noises, diags_inits, BC_stds)
    prods = [p for p in prod1]
    prods += [p for p in prod2]
    i = 0
    for optimizer, lag, lr, noise, diags_init, BC_std in prods:
        i += 1
        print(i, len(prods))
        model = get_full_ssm(num_hidden_state=hidden_size,
                                        device=device,
                                        A_init_func=lambda n: diags_init * torch.eye(n) + torch.randn([n, n]) * noise,
                                        B_init_std=BC_std,
                                        C_init_std=BC_std
                             )

        saving_path = get_saving_path_for_exp(optimizer, lag, lr, noise, diags_init, BC_std)

        print(os.path.basename(saving_path))
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = optimizer(model.parameters(),
                                  lr=lr)
            train_smm_over_white_noise_lag_multiprocess(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                early_stop=False,
                min_cut=epochs
            )

def grid_search_hippo():
    def get_saving_path_for_exp(optimizer, lag, lr, dt):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplemented()

        return os.path.join("../..", "results", "grid",
                            "hippo_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                dt=dt,
                                hidden=hidden_size
                            ))

    dts = [1, 0.5, 0.1, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.0001, 0.00005]

    prod1 = product([torch.optim.SGD], lags, lrs_sgd, dts)
    prod2 = product([torch.optim.Adam], lags, lrs_adam, dts)
    prods = [p for p in prod1]
    prods += [p for p in prod2]
    i = 0
    for optimizer, lag, lr, dt in prods:
        i += 1
        print(i, len(prods))
        model = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                         device=device,
                                         dt=dt,
                                         disc_only_onces=True)

        saving_path = get_saving_path_for_exp(optimizer, lag, lr, dt)

        print(os.path.basename(saving_path))
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = optimizer(model.parameters(),
                                  lr=lr)
            train_smm_over_white_noise_lag_multiprocess(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                early_stop=False,
                min_cut=epochs
            )


def grid_search_hippo_no_a():
    def get_saving_path_for_exp(optimizer, lag, lr, dt, no):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplemented()

        return os.path.join("../..", "results", "grid",
                            "hippo_no{no}_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                dt=dt,
                                no=no,
                                hidden=hidden_size
                            ))

    dts = [1, 0.5, 0.1, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.0001, 0.00005]
    nos = ["a", "ac"]
    prod1 = product([torch.optim.SGD], lags, lrs_sgd, dts, nos)
    prod2 = product([torch.optim.Adam], lags, lrs_adam, dts, nos)
    prods = [p for p in prod1]
    prods += [p for p in prod2]
    i = 0
    for optimizer, lag, lr, dt, no in prods:
        i += 1
        print(i, len(prods))
        if no == "a":
            model = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                              device=device,
                                              dt=dt,
                                              disc_only_onces=True,
                                              trainable_param_list=["B","C"])

        elif no == "ac":
            model = get_full_disc_hippo_model(num_hidden_state=hidden_size,
                                              device=device,
                                              dt=dt,
                                              disc_only_onces=True,
                                              trainable_param_list=["C"])
        else:
            raise NotImplemented

        saving_path = get_saving_path_for_exp(optimizer, lag, lr, dt, no)

        print(os.path.basename(saving_path))
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = optimizer(model.parameters(),
                                  lr=lr)
            train_smm_over_white_noise_lag_multiprocess(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                early_stop=False,
                min_cut=epochs
            )

def grid_search_rot():
    def get_saving_path_for_exp(optimizer, lag, lr, rot_type):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplemented()

        return os.path.join("../..", "results", "grid",
                            "rot_{rot_type}_h{hidden}_l{lag}_lr{lr}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                rot_type=rot_type,
                                hidden=hidden_size
                            ))

    rot_types = ["one_over", "eq"]

    prod1 = product([torch.optim.SGD], lags, lrs_sgd, rot_types)
    prod2 = product([torch.optim.Adam], lags, lrs_adam, rot_types)
    prods = [p for p in prod1]
    prods += [p for p in prod2]
    i = 0
    for optimizer, lag, lr, rot_type in prods:
        i += 1
        print(i, len(prods))

        if rot_type == "one_over":
            model = get_rot_ssm_one_over_n(num_hidden_state=hidden_size,
                                             device=device)
        elif rot_type == "eq":
            model = get_rot_ssm_equally_spaced(num_hidden_state=hidden_size,
                                             device=device)
        else:
            raise NotImplemented()

        saving_path = get_saving_path_for_exp(optimizer, lag, lr, rot_type)

        print(os.path.basename(saving_path))
        with SSMTrainingLogger(saving_path=saving_path) as logger:
            optimizer = optimizer(model.parameters(),
                                  lr=lr)
            train_smm_over_white_noise_lag_multiprocess(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=optimizer,
                early_stop=False,
                min_cut=epochs
            )

if __name__ == "__main__":
    #grid_search_fssm_diag()
    #grid_search_hippo()
    grid_search_hippo_no_a()

