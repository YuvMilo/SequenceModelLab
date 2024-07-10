import os
import torch
import ray
from itertools import product

from src.multiprocessing.utils import ProgressBar
from src.experiments.utils.get_standard_tasks_ssm import get_ssm_training_task, \
    get_hippo_training_task, get_rot_training_task

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_param_every = 16
save_dir = os.path.join("..", "results", "non_linear", "grid")
hidden_size = 128
lags = [64]
seq_len = hidden_size * 2
epochs = 4000
batch_size = 128


optimizers = [torch.optim.Adam]
lrs_adam = [0.1, 0.01, 0.001, 0.0001]


def get_tasks_fssm(progress_bar_actor,
                   log_param_freq: int = 1):

    diags_inits = [0.0, 0.9, 1, 1.1, 2]
    noises = [0.0001, 0.001, 0.01]
    BC_stds = [1, 0.1, 0.01, 0.001]

    prods = product([torch.optim.Adam], lags, lrs_adam, noises, diags_inits, BC_stds)
    tasks = []
    for _, lag, lr, noise, diags_init, BC_std in prods:
        res = {
            "lr": lr,
            "noise": noise,
            "diag_init": diags_init,
            "BC_std": BC_std,
            "opt": 'adam',
            "model_name": "ssm",
        }

        task = get_ssm_training_task(
            training_dict=res,
            lag=lag,
            hidden_size=hidden_size,
            epochs=epochs,
            seq_len=seq_len,
            progress_bar_actor=progress_bar_actor,
            save_dir=save_dir,
            device=device,
            training_uuid="",
            log_param_every=log_param_freq
        )
        tasks.append(task)
    return tasks


def get_tasks_hippo(progress_bar_actor,
                    log_param_freq: int = 1):
    dts = [1, 0.5, 0.1, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.0001, 0.00005]
    prods = product([torch.optim.Adam], lags, lrs_adam, dts)
    tasks = []
    for _, lag, lr, dt in prods:
        res = {
            "lr": lr,
            "dt": dt,
            "opt": 'adam',
            "model_name": "hippo",
        }

        task = get_hippo_training_task(
            training_dict=res,
            lag=lag,
            hidden_size=hidden_size,
            epochs=epochs,
            seq_len=seq_len,
            progress_bar_actor=progress_bar_actor,
            save_dir=save_dir,
            device=device,
            training_uuid="",
            log_param_every=log_param_freq
        )
        tasks.append(task)
    return tasks


def get_tasks_rot(progress_bar_actor,
                  log_param_freq: int = 1):
    rot_types = ["one_over", "eq"]
    prods = product([torch.optim.Adam], lags, lrs_adam, rot_types)
    tasks = []
    for _, lag, lr, rot_type in prods:
        res = {
            "lr": lr,
            "rot_type": rot_type,
            "opt": 'adam',
            "model_name": "hippo_" + rot_type,
        }

        task = get_rot_training_task(
            training_dict=res,
            lag=lag,
            hidden_size=hidden_size,
            epochs=epochs,
            seq_len=seq_len,
            progress_bar_actor=progress_bar_actor,
            save_dir=save_dir,
            device=device,
            training_uuid="",
            log_param_every=log_param_freq
        )
        tasks.append(task)
    return tasks


def run_exp():

    ray.init(ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    tasks = []
    tasks += get_tasks_rot(progress_bar_actor,
                           log_param_freq=log_param_every)
    tasks += get_tasks_hippo(progress_bar_actor,
                             log_param_freq=log_param_every)
    tasks += get_tasks_fssm(progress_bar_actor,
                            log_param_freq=log_param_every)

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    a = prods = product([torch.optim.Adam], lags, lrs_adam, ['1'])
    pass
