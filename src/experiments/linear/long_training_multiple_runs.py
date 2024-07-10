import os
import torch
import ray
import numpy as np

from src.multiprocessing.utils import ProgressBar
from src.experiments.utils.best_settings import top_hippo_res_linear, \
    top_rot_res_linear, top_hippo_no_res_linear, top_fssm_res_linear
from src.experiments.utils.get_standard_tasks_linear_ssm import get_ssm_training_task, \
    get_hippo_training_task, get_hippo_no_training_task, get_rot_training_task

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_param_every = 16
save_dir = os.path.join("..", "results", "variance_mult")
hidden_sizes = [64]
seq_len = np.max(hidden_sizes) * 2
epochs = 40000

NUM_TIMES = 10


def get_tasks_fssm(progress_bar_actor,
                   log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for res in top_fssm_res_linear:
            for i in range(NUM_TIMES):
                training_uuid = str(i)
                task = get_ssm_training_task(
                    training_dict=res,
                    lag=lag,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    seq_len=seq_len,
                    progress_bar_actor=progress_bar_actor,
                    save_dir=save_dir,
                    device=device,
                    training_uuid=training_uuid,
                    log_param_every=log_param_freq
                )
                tasks.append(task)
    return tasks


def get_tasks_hippo(progress_bar_actor,
                    log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for res in top_hippo_res_linear:
            for i in range(NUM_TIMES):
                training_uuid = str(i)
                task = get_hippo_training_task(
                    training_dict=res,
                    lag=lag,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    seq_len=seq_len,
                    progress_bar_actor=progress_bar_actor,
                    save_dir=save_dir,
                    device=device,
                    training_uuid=training_uuid,
                    log_param_every=log_param_freq
                )
                tasks.append(task)
    return tasks


def get_tasks_hippo_no_a(progress_bar_actor,
                         log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for res in top_hippo_no_res_linear:
            for i in range(NUM_TIMES):
                training_uuid = str(i)
                task = get_hippo_no_training_task(
                    training_dict=res,
                    lag=lag,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    seq_len=seq_len,
                    progress_bar_actor=progress_bar_actor,
                    save_dir=save_dir,
                    device=device,
                    training_uuid=training_uuid,
                    log_param_every=log_param_freq
                )
                tasks.append(task)
    return tasks


def get_tasks_rot(progress_bar_actor,
                  log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for res in top_rot_res_linear:
            for i in range(NUM_TIMES):
                training_uuid = str(i)
                task = get_rot_training_task(
                    training_dict=res,
                    lag=lag,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    seq_len=seq_len,
                    progress_bar_actor=progress_bar_actor,
                    save_dir=save_dir,
                    device=device,
                    training_uuid=training_uuid,
                    log_param_every=log_param_freq
                )
                tasks.append(task)
    return tasks


def run_exp():
    ray.init(ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    tasks = []
    tasks += get_tasks_hippo_no_a(progress_bar_actor,
                                  log_param_freq=log_param_every)
    tasks += get_tasks_rot(progress_bar_actor,
                           log_param_freq=log_param_every)
    tasks += get_tasks_hippo(progress_bar_actor,
                             log_param_freq=log_param_every)
    tasks += get_tasks_fssm(progress_bar_actor,
                            log_param_freq=log_param_every)

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    run_exp()
