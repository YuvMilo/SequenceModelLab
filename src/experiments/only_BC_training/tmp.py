import torch
import numpy as np
import ray

import os

from src.experiments.utils.get_standard_task_saving_str import\
    get_noise_training_task_file_name, get_noise_complex_diag_training_task_file_name, \
    get_rot_training_task_file_name
from src.experiments.utils.misc import opt_str_to_opt
from src.models.factories.classical_ssm_factories import get_flexible_ssm, \
    get_flexible_complex_diag_ssm, get_rot_ssm_equally_spaced
from src.training.train_ssm import train_smm_over_white_noise_lag_multiprocess
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.algorithms.misc import matrix_to_real_2x2block_matrix_with_same_eigenvalues
from src.multiprocessing.utils import ProgressBar
from src.models.strategies.storing import BlockDiagStoringStrategy

DEFAULT_SAMPLE_NUMS = 128
NOMALIZED_MAX_RADII = 0.93


def get_block_diag_vs_complex_training_tasks(
        training_dict: dict, lag: int, hidden_size: int, epochs: int,
        seq_len: int,
        progress_bar_actor,
        save_dir: str,
        device: torch.device,
        training_uuid: str = "",
        log_param_every: int = 1,
        samples_num: int = DEFAULT_SAMPLE_NUMS
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["samples_num"] = samples_num
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    BC_std = float(training_dict["BC_std"])

    A = torch.randn([hidden_size, hidden_size]) * 1 / np.sqrt(hidden_size)
    B = torch.ones([hidden_size, 1], dtype=torch.float)
    C = BC_std * torch.randn([1, hidden_size], dtype=torch.float)
    D = torch.zeros([1, 1], dtype=torch.float)

    max_A_eigs_norm = torch.max(torch.abs(torch.linalg.eig(A)[0]))
    A = A / max_A_eigs_norm * NOMALIZED_MAX_RADII

    def init_noise(n, input_dim=1, output_dim=1):
        return A.clone(), B.clone(), C.clone(), D.clone()

    def init_noise_diag(n, input_dim=1, output_dim=1):
        return matrix_to_real_2x2block_matrix_with_same_eigenvalues(A.clone()), \
            B.clone(), C.clone(), D.clone()

    def init_noise_complex_diag_only_A(n, input_dim=1, output_dim=1):
        eig, _ = torch.linalg.eig(A)
        return eig, B.clone(), C.clone(), D.clone()

    model_noise = get_flexible_ssm(init_func=init_noise,
                                   num_hidden_state=hidden_size,
                                   device=device,
                                   trainable_param_list=["C", "B"])

    model_block_diag = get_flexible_ssm(init_func=init_noise_diag,
                                        num_hidden_state=hidden_size,
                                        device=device,
                                        ssm_storing_strategy=BlockDiagStoringStrategy,
                                        trainable_param_list=["C", "B"])

    # model_block_diag_init = get_flexible_ssm(init_func=init_noise_diag,
    #                                          num_hidden_state=hidden_size,
    #                                          device=device,
    #                                          trainable_param_list=["C","B"])

    model_complex_only_A = get_flexible_complex_diag_ssm(
        init_func=init_noise_complex_diag_only_A,
        num_hidden_state=hidden_size,
        device=device,
        only_A_is_complex=True,
        trainable_param_list=["C", "B"]
    )

    noise_saving_path = get_noise_training_task_file_name(save_dir=save_dir,
                                                          optimizer=opt, lag=lag, lr=lr,
                                                          BC_std=BC_std,
                                                          training_uuid=training_uuid,
                                                          hidden_size=hidden_size,
                                                          is_diagonal=False)

    # block_diag_init_saving_path = get_noise_training_task_file_name(
    #     save_dir=save_dir,
    #     optimizer=opt, lag=lag, lr=lr,
    #     BC_std=BC_std,
    #     training_uuid=training_uuid,
    #     hidden_size=hidden_size,
    #     is_diagonal_init=True
    # )

    block_diag_saving_path = get_noise_training_task_file_name(
        save_dir=save_dir,
        optimizer=opt, lag=lag, lr=lr,
        BC_std=BC_std,
        training_uuid=training_uuid,
        hidden_size=hidden_size,
        is_diagonal=True
    )

    complex_only_A_saving_path = get_noise_complex_diag_training_task_file_name(
        save_dir=save_dir,
        optimizer=opt, lag=lag, lr=lr,
        BC_std=BC_std,
        training_uuid=training_uuid,
        hidden_size=hidden_size,
        only_A=True
    )

    training_dict_noise = training_dict.copy()
    training_dict_noise["is_diagonal"] = False
    training_dict_noise["is_diagonal_init"] = False
    training_dict_noise["is_complex"] = False
    training_dict_noise["onlyA"] = False
    training_dict_noise["model_name"] = "noise"

    # training_dict_block_diagonal_init = training_dict.copy()
    # training_dict_block_diagonal_init["is_diagonal"] = False
    # training_dict_block_diagonal_init["is_diagonal_init"] = True
    # training_dict_block_diagonal_init["is_complex"] = False
    # training_dict_block_diagonal_init["onlyA"] = False
    # training_dict_block_diagonal_init["model_name"] = "block_diag_init"

    training_dict_block_diagonal = training_dict.copy()
    training_dict_block_diagonal["is_diagonal"] = True
    training_dict_block_diagonal["is_diagonal_init"] = True
    training_dict_block_diagonal["is_complex"] = False
    training_dict_block_diagonal["model_name"] = "block_diag"

    training_dict_complex_only_A = training_dict.copy()
    training_dict_complex_only_A["is_diagonal"] = True
    training_dict_complex_only_A["is_diagonal_init"] = True
    training_dict_complex_only_A["is_complex"] = True
    training_dict_complex_only_A["onlyA"] = True
    training_dict_complex_only_A["model_name"] = "complex_only_A"

    logger_noise = SSMTrainingLogger(saving_path=noise_saving_path,
                                     running_params=training_dict_noise,
                                     param_storing_freq=log_param_every)

    logger_block_diag = SSMTrainingLogger(saving_path=block_diag_saving_path,
                                          running_params=training_dict_block_diagonal,
                                          param_storing_freq=log_param_every)

    logger_complex_only_A = SSMTrainingLogger(saving_path=complex_only_A_saving_path,
                                              running_params=training_dict_complex_only_A,
                                              param_storing_freq=log_param_every)

    tasks = []

    tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
        model=model_noise,
        lag=lag,
        seq_len=seq_len,
        logger=logger_noise,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor
    ))

    # tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
    #     model=model_block_diag_init,
    #     lag=lag,
    #     seq_len=seq_len,
    #     logger=logger_block_diag_init,
    #     num_epochs=epochs,
    #     optimizer=opt,
    #     lr=lr,
    #     early_stop=False,
    #     min_cut=epochs,
    #     progress_bar_actor=progress_bar_actor
    # ))

    tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
        model=model_block_diag,
        lag=lag,
        seq_len=seq_len,
        logger=logger_block_diag,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor
    ))

    tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
        model=model_complex_only_A,
        lag=lag,
        seq_len=seq_len,
        logger=logger_complex_only_A,
        num_epochs=epochs,
        optimizer=opt,
        early_stop=False,
        min_cut=epochs,
        lr=lr,
        progress_bar_actor=progress_bar_actor
    ))

    return tasks


def get_ssm_rot_block_diag_vs_complex(
        training_dict: dict, lag: int, hidden_size: int, epochs: int,
        seq_len: int,
        progress_bar_actor,
        save_dir: str,
        device: torch.device,
        training_uuid: str = "",
        log_param_every: int = 1,
        samples_num: int = DEFAULT_SAMPLE_NUMS
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["samples_num"] = samples_num
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    BC_std = float(training_dict["BC_std"])

    rot_block_diag_model = get_rot_ssm_equally_spaced(
        num_hidden_state=hidden_size,
        device=device,
        radii=NOMALIZED_MAX_RADII,
        main_diagonal_diff=0,
        off_diagonal_ratio=1,
        B_init_std=BC_std,
        C_init_std=BC_std,
        block_diag=True
    )

    A, _, C, D = rot_block_diag_model.get_params()
    B = torch.ones([hidden_size, 1], dtype=torch.float)
    A, C, D = A.clone().detach(), C.clone().detach(), D.clone().detach()

    def init_noise_diag(n, input_dim=1, output_dim=1):
        return matrix_to_real_2x2block_matrix_with_same_eigenvalues(A.clone()), \
            B.clone(), C.clone(), D.clone()

    def init_noise_complex_diag_only_A(n, input_dim=1, output_dim=1):
        eig, _ = torch.linalg.eig(A)
        return eig, B.clone(), C.clone(), D.clone()

    rot_block_diag_model = get_flexible_ssm(init_func=init_noise_diag,
                                            num_hidden_state=hidden_size,
                                            device=device,
                                            ssm_storing_strategy=BlockDiagStoringStrategy,
                                            trainable_param_list=["C", "B"])

    # rot_block_diag_init_model = get_flexible_ssm(init_func=init_noise_diag,
    #                                         num_hidden_state=hidden_size,
    #                                         device=device,
    #                                         trainable_param_list=["C", "B"])

    model_complex_only_A = get_flexible_complex_diag_ssm(
        init_func=init_noise_complex_diag_only_A,
        num_hidden_state=hidden_size,
        device=device,
        only_A_is_complex=True,
        trainable_param_list=["C", "B"]
    )

    # rot_block_diag_init_saving_path = get_rot_training_task_file_name(
    #     save_dir=save_dir,
    #     optimizer=opt, lag=lag, lr=lr,
    #     training_uuid=training_uuid,
    #     hidden_size=hidden_size,
    #     main_diagonal_diff=0,
    #     off_diagonal_ratio=1,
    #     rot_type="eq_block_diag_init"
    # )

    rot_block_diag_saving_path = get_rot_training_task_file_name(
        save_dir=save_dir,
        optimizer=opt, lag=lag, lr=lr,
        training_uuid=training_uuid,
        hidden_size=hidden_size,
        main_diagonal_diff=0,
        off_diagonal_ratio=1,
        rot_type="eq_block_diag"
    )

    rot_complex_diag_saving_path = get_rot_training_task_file_name(
        save_dir=save_dir,
        optimizer=opt, lag=lag, lr=lr,
        training_uuid=training_uuid,
        hidden_size=hidden_size,
        main_diagonal_diff=0,
        off_diagonal_ratio=1,
        rot_type="eq_complex_diag_only_A"
    )

    # training_dict_block_diagonal_init = training_dict.copy()
    # training_dict_block_diagonal_init["is_diagonal_init"] = True
    # training_dict_block_diagonal_init["is_diagonal"] = False
    # training_dict_block_diagonal_init["is_complex"] = False
    # training_dict_block_diagonal_init["model_name"] = "rot_block_diag_init"

    training_dict_block_diagonal = training_dict.copy()
    training_dict_block_diagonal["is_diagonal_init"] = True
    training_dict_block_diagonal["is_diagonal"] = True
    training_dict_block_diagonal["is_complex"] = False
    training_dict_block_diagonal["model_name"] = "rot_block_diag"

    training_dict_complex_only_A = training_dict.copy()
    training_dict_block_diagonal["is_diagonal_init"] = True
    training_dict_complex_only_A["is_diagonal"] = True
    training_dict_complex_only_A["is_complex"] = True
    training_dict_complex_only_A["onlyA"] = True
    training_dict_complex_only_A["model_name"] = "rot_complex_only_A"

    # logger_block_diag_init = SSMTrainingLogger(
    #     saving_path=rot_block_diag_init_saving_path,
    #     running_params=training_dict_block_diagonal_init,
    #     param_storing_freq=log_param_every
    # )

    logger_block_diag = SSMTrainingLogger(saving_path=rot_block_diag_saving_path,
                                          running_params=training_dict_block_diagonal,
                                          param_storing_freq=log_param_every)

    logger_complex_only_A = SSMTrainingLogger(saving_path=rot_complex_diag_saving_path,
                                              running_params=training_dict_complex_only_A,
                                              param_storing_freq=log_param_every)

    # task_block_diag_init = train_smm_over_white_noise_lag_multiprocess.remote(
    #     model=rot_block_diag_init_model,
    #     lag=lag,
    #     seq_len=seq_len,
    #     logger=logger_block_diag_init,
    #     num_epochs=epochs,
    #     optimizer=opt,
    #     early_stop=False,
    #     min_cut=epochs,
    #     lr=lr,
    #     progress_bar_actor=progress_bar_actor
    # )

    task_block_diag = train_smm_over_white_noise_lag_multiprocess.remote(
        model=rot_block_diag_model,
        lag=lag,
        seq_len=seq_len,
        logger=logger_block_diag,
        num_epochs=epochs,
        optimizer=opt,
        early_stop=False,
        min_cut=epochs,
        lr=lr,
        progress_bar_actor=progress_bar_actor
    )

    task_complex_diag = train_smm_over_white_noise_lag_multiprocess.remote(
        model=model_complex_only_A,
        lag=lag,
        seq_len=seq_len,
        logger=logger_complex_only_A,
        num_epochs=epochs,
        optimizer=opt,
        early_stop=False,
        min_cut=epochs,
        lr=lr,
        progress_bar_actor=progress_bar_actor
    )

    return [task_block_diag, task_complex_diag]


def run_exp():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

    ray.init(num_cpus=16, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    training_dict = {'lr': '0.001',
                     'BC_std': '0.1',
                     'opt': 'adam'}

    # hidden = 64
    times = 4
    hiddens = [64, 128]
    max_lag = np.max(hiddens) * 2
    tasks = []
    for i in range(times):
        for hidden in hiddens:
            savedir = os.path.join("..", "results",
                                   "only_BC_trained", str(hidden))
            jumps = int(max_lag / 16)
            for lag in range(16, max_lag, jumps):
                tasks += get_ssm_rot_block_diag_vs_complex(
                    training_dict=training_dict.copy(),
                    lag=lag,
                    hidden_size=hidden,
                    epochs=3000,
                    seq_len=300,
                    save_dir=savedir,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    training_uuid=i,
                    log_param_every=4,
                    samples_num=64,
                    progress_bar_actor=progress_bar_actor
                )

                tasks += get_block_diag_vs_complex_training_tasks(
                    training_dict=training_dict.copy(),
                    lag=lag,
                    hidden_size=hidden,
                    epochs=3000,
                    seq_len=300,
                    save_dir=savedir,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    training_uuid=i,
                    log_param_every=4,
                    samples_num=64,
                    progress_bar_actor=progress_bar_actor
                )

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    run_exp()
