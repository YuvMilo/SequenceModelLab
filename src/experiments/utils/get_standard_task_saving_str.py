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
                                    main_diagonal_diff: float = 1,
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
