import torch.optim
import torch

from src.experiments.utils.misc import opt_str_to_opt
from src.models.factories.classical_ssm_factories import get_rot_ssm_one_over_n, \
    get_rot_ssm_equally_spaced, get_full_ssm_hippo1D, get_full_ssm_diag_plus_noise1D
from src.training.train_ssm import train_ssm_multiprocess
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.experiments.utils.get_standard_task_saving_str import \
    get_hippo_no_training_task_file_name, get_ssm_training_task_file_name, \
    get_rot_training_task_file_name, get_hippo_training_task_file_name
from src.multiprocessing.utils import ProgressBarActor

non_linearity = torch.tanh
DEFAULT_SAMPLE_NUMS = 128


def get_ssm_training_task(training_dict: dict, lag: int, hidden_size: int, epochs: int,
                          seq_len: int,
                          progress_bar_actor,
                          save_dir: str,
                          device: torch.device,
                          training_uuid: str = "",
                          log_param_every: int = 1,
                          samples_num: int = DEFAULT_SAMPLE_NUMS
                          ):
    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["samples_num"] = samples_num
    training_dict["non_linearity"] = str(non_linearity)
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    noise = float(training_dict["noise"])
    diags_init = float(training_dict["diag_init"])
    BC_std = float(training_dict["BC_std"])
    model = get_full_ssm_diag_plus_noise1D(
        num_hidden_state=hidden_size,
        device=device,
        B_init_std=BC_std,
        C_init_std=BC_std,
        A_noise_std=noise,
        A_diag=diags_init,
        non_linearity=non_linearity
    )
    saving_path = get_ssm_training_task_file_name(save_dir=save_dir,
                                                  optimizer=opt, lag=lag, lr=lr,
                                                  noise=noise, diags_init=diags_init,
                                                  BC_std=BC_std,
                                                  training_uuid=training_uuid,
                                                  hidden_size=hidden_size)

    logger = SSMTrainingLogger(saving_path=saving_path,
                               running_params=training_dict,
                               param_storing_freq=log_param_every)
    task = train_ssm_multiprocess.remote(
        model=model,
        lag=lag,
        seq_len=seq_len,
        samples_num=samples_num,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor,
    )
    return task


def get_hippo_training_task(training_dict: dict, lag: int, hidden_size: int,
                            epochs: int, seq_len: int,
                            progress_bar_actor: ProgressBarActor,
                            save_dir: str,
                            device: torch.device,
                            training_uuid: str = "",
                            log_param_every: int = 1,
                            samples_num: int = DEFAULT_SAMPLE_NUMS
                            ):
    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["samples_num"] = samples_num
    training_dict["non_linearity"] = str(non_linearity)
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    dt = float(training_dict["dt"])

    model = get_full_ssm_hippo1D(num_hidden_state=hidden_size,
                                 device=device,
                                 dt=dt,
                                 non_linearity=non_linearity)

    saving_path = get_hippo_training_task_file_name(save_dir=save_dir,
                                                    optimizer=opt, lag=lag, lr=lr,
                                                    dt=dt, training_uuid=training_uuid,
                                                    hidden_size=hidden_size)

    logger = SSMTrainingLogger(saving_path=saving_path,
                               running_params=training_dict,
                               param_storing_freq=log_param_every)
    task = train_ssm_multiprocess.remote(
        model=model,
        lag=lag,
        seq_len=seq_len,
        samples_num=samples_num,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor,
    )
    return task


def get_hippo_no_training_task(training_dict: dict, lag: int, hidden_size: int,
                               epochs: int, seq_len: int,
                               progress_bar_actor, save_dir: str,
                               device: torch.device,
                               training_uuid: str = "",
                               log_param_every: int = 1,
                               samples_num: int = DEFAULT_SAMPLE_NUMS
                               ):

    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["samples_num"] = samples_num
    training_dict["non_linearity"] = str(non_linearity)
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    dt = float(training_dict["dt"])
    no = training_dict["no"]

    if no == "a":
        trainable_param_list = ["C", "B"]

    elif no == "ac":
        trainable_param_list = ["B"]
    else:
        raise NotImplementedError

    model = get_full_ssm_hippo1D(num_hidden_state=hidden_size,
                                 device=device,
                                 dt=dt,
                                 trainable_param_list=trainable_param_list,
                                 non_linearity=non_linearity)

    saving_path = get_hippo_no_training_task_file_name(save_dir=save_dir,
                                                       optimizer=opt, lag=lag, lr=lr,
                                                       no=no, dt=dt,
                                                       training_uuid=training_uuid,
                                                       hidden_size=hidden_size)

    logger = SSMTrainingLogger(saving_path=saving_path,
                               running_params=training_dict,
                               param_storing_freq=log_param_every)

    task = train_ssm_multiprocess.remote(
        model=model,
        lag=lag,
        seq_len=seq_len,
        samples_num=samples_num,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor,
    )
    return task


def get_rot_training_task(training_dict: dict, lag: int, hidden_size: int,
                          epochs: int, seq_len: int,
                          progress_bar_actor, save_dir: str,
                          device: torch.device,
                          training_uuid: str = "",
                          main_diagonal_diff: float = 0,
                          off_diagonal_ratio: float = 1,
                          log_param_every: int = 1,
                          samples_num: int = DEFAULT_SAMPLE_NUMS):

    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["main_diagonal_diff"] = main_diagonal_diff
    training_dict["off_diagonal_ratio"] = off_diagonal_ratio
    training_dict["samples_num"] = samples_num
    training_dict["non_linearity"] = str(non_linearity)
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    rot_type = training_dict["rot_type"]

    if rot_type == "one_over":
        model = get_rot_ssm_one_over_n(num_hidden_state=hidden_size,
                                       device=device,
                                       main_diagonal_diff=main_diagonal_diff,
                                       off_diagonal_ratio=off_diagonal_ratio,
                                       non_linearity=non_linearity)
    elif rot_type == "eq":
        model = get_rot_ssm_equally_spaced(num_hidden_state=hidden_size,
                                           device=device,
                                           main_diagonal_diff=main_diagonal_diff,
                                           off_diagonal_ratio=off_diagonal_ratio,
                                           non_linearity=non_linearity)
    else:
        raise NotImplementedError()

    saving_path = get_rot_training_task_file_name(save_dir=save_dir,
                                                  optimizer=opt, lag=lag, lr=lr,
                                                  rot_type=rot_type,
                                                  training_uuid=training_uuid,
                                                  hidden_size=hidden_size,
                                                  main_diagonal_diff=main_diagonal_diff,
                                                  off_diagonal_ratio=off_diagonal_ratio
                                                  )

    logger = SSMTrainingLogger(saving_path=saving_path,
                               running_params=training_dict,
                               param_storing_freq=log_param_every)

    task = train_ssm_multiprocess.remote(
        model=model,
        lag=lag,
        seq_len=seq_len,
        samples_num=samples_num,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        progress_bar_actor=progress_bar_actor,
    )
    return task
