import os
import torch
from src.experiments.utils.misc import opt_to_opt_str


def get_ssm_training_task_file_name(save_dir: str, optimizer: torch.optim.Optimizer,
                                    lag: int, lr: float, noise: float,
                                    diags_init: float, BC_std: float,
                                    training_uuid: str, hidden_size: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "ssm_h{hidden}_l{lag}_lr{lr}_n{noise}_d{diag_init}_b{BC_std}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            noise=noise,
                            diag_init=diags_init,
                            BC_std=BC_std,
                            opt=opt_str,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))


def get_hippo_training_task_file_name(save_dir: str, optimizer: torch.optim.Optimizer,
                                      lag: int, lr: float, dt: float,
                                      training_uuid: str, hidden_size: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "hippo_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            dt=dt,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))


def get_rot_training_task_file_name(save_dir: str, optimizer: torch.optim.Optimizer,
                                    lag: int, lr: float, rot_type: str,
                                    training_uuid: str, hidden_size: int,
                                    main_diagonal_diff: float = 0,
                                    off_diagonal_ratio: float = 1
                                    ) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "rot_h{hidden}_{rot_type}_l{lag}_lr{lr}_o{opt}_id{training_uuid}_dd{diag_diff}_or_{off}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            rot_type=rot_type,
                            training_uuid=training_uuid,
                            hidden=hidden_size,
                            diag_diff=main_diagonal_diff,
                            off=off_diagonal_ratio
                        ))


def get_hippo_no_training_task_file_name(save_dir: str,
                                         optimizer: torch.optim.Optimizer,
                                         lag: int, lr: float, dt: float,
                                         no: str, training_uuid: str,
                                         hidden_size: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "hippo_no{no}_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            dt=dt,
                            no=no,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))


def get_deep_ssm_training_task_file_name(save_dir: str,
                                         optimizer: torch.optim.Optimizer,
                                         lag: int, lr: float, noise: float,
                                         diags_init: float, BC_std: float,
                                         training_uuid: str, hidden_size: int,
                                         depth: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "deepssm_d{depth}_h{hidden}_l{lag}_lr{lr}_n{noise}_d{diag_init}_b{BC_std}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            noise=noise,
                            diag_init=diags_init,
                            BC_std=BC_std,
                            opt=opt_str,
                            training_uuid=training_uuid,
                            hidden=hidden_size,
                            depth=depth
                        ))


def get_deep_hippo_training_task_file_name(save_dir: str,
                                           optimizer: torch.optim.Optimizer,
                                           lag: int, lr: float, dt: float,
                                           training_uuid: str, hidden_size: int,
                                           depth: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "deephippo_d{depth}_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            dt=dt,
                            training_uuid=training_uuid,
                            hidden=hidden_size,
                            depth=depth
                        ))


def get_deep_rot_training_task_file_name(
        save_dir: str, optimizer: torch.optim.Optimizer,
        lag: int, lr: float, rot_type: str,
        training_uuid: str, hidden_size: int,
        depth: int,
        main_diagonal_diff: float = 0,
        off_diagonal_ratio: float = 1,

) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "deeprot_d{depth}_h{hidden}_{rot_type}_l{lag}_lr{lr}_o{opt}_id{training_uuid}_dd{diag_diff}_or_{off}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            rot_type=rot_type,
                            training_uuid=training_uuid,
                            hidden=hidden_size,
                            diag_diff=main_diagonal_diff,
                            off=off_diagonal_ratio,
                            depth=depth
                        ))


def get_deephippo_no_training_task_file_name(save_dir: str,
                                             optimizer: torch.optim.Optimizer,
                                             lag: int, lr: float, dt: float,
                                             no: str, training_uuid: str,
                                             hidden_size: int,
                                             depth: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "deephippo_no{no}_d{depth}_h{hidden}_l{lag}_lr{lr}_dt{dt}_o{opt}_id{training_uuid}".format(
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            dt=dt,
                            no=no,
                            training_uuid=training_uuid,
                            hidden=hidden_size,
                            depth=depth
                        ))


def get_noise_training_task_file_name(save_dir: str, optimizer: torch.optim.Optimizer,
                                      lag: int, lr: float,
                                      BC_std: float,
                                      training_uuid: str, hidden_size: int,
                                      is_diagonal: bool = False,
                                      is_diagonal_init: bool = False) -> str:
    opt_str = opt_to_opt_str[optimizer]

    diagonal_str = "noise"
    if is_diagonal:
        diagonal_str = "diagonal"
    elif is_diagonal_init:
        diagonal_str = "diagonal_init"

    return os.path.join(save_dir,
                        "ssm_{digonal_str}_h{hidden}_l{lag}_lr{lr}_b{BC_std}_o{opt}_id{training_uuid}".format(
                            digonal_str=diagonal_str,
                            lag=lag,
                            lr=lr,
                            BC_std=BC_std,
                            opt=opt_str,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))


def get_shift_training_task_file_name(save_dir: str, optimizer: torch.optim.Optimizer,
                                      lag: int, lr: float,
                                      radii: float,
                                      training_uuid: str, hidden_size: int) -> str:
    opt_str = opt_to_opt_str[optimizer]

    return os.path.join(save_dir,
                        "ssm_r{radii}_h{hidden}_l{lag}_lr{lr}_o{opt}_id{training_uuid}".format(
                            radii=radii,
                            lag=lag,
                            lr=lr,
                            opt=opt_str,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))


def get_noise_complex_diag_training_task_file_name(
        save_dir: str, optimizer: torch.optim.Optimizer,
        lag: int, lr: float,
        BC_std: float,
        training_uuid: str, hidden_size: int,
        only_A: bool
) -> str:
    opt_str = opt_to_opt_str[optimizer]

    complex_str = "all"
    if only_A:
        complex_str = "onlyA"

    return os.path.join(save_dir,
                        "ssm_complex_{complex_str}_diag_h{hidden}_l{lag}_lr{lr}_b{BC_std}_o{opt}_id{training_uuid}".format(
                            complex_str=complex_str,
                            lag=lag,
                            lr=lr,
                            BC_std=BC_std,
                            opt=opt_str,
                            training_uuid=training_uuid,
                            hidden=hidden_size
                        ))
