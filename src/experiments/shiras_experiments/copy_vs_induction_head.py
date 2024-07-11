import os
import torch
import ray
import numpy as np

import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to sys.path
sys.path.append(os.path.join(current_dir, '../../../'))

from src.multiprocessing.utils import ProgressBar
from src.experiments.utils.best_settings import top_deep_rot_res, \
    top_deep_fssm_res
from src.experiments.utils.get_standard_tasks_deep import get_ssm_training_task, \
    get_rot_training_task

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_param_every = 8
save_dir = os.path.join("..", "results", "deep", "changing_lag")
hidden_sizes = [32] # [32, 64, 96, 128, 192]
seq_len = np.max(hidden_sizes) * 2
epochs = 200 # 20000

NUM_TIMES = 1 # 10


# delay task
def get_tasks_fssm(progress_bar_actor,
                   log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for res in top_deep_fssm_res:
            for i in range(NUM_TIMES):
                depth = res["depth"]
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
                    log_param_every=log_param_freq,
                    depth=depth
                )
                if task is not None:
                    tasks.append(task)
    return tasks


# induction head task
def get_tasks_induction_head(progress_bar_actor,
                  log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        target=[0] # correct this... 
        for res in top_deep_rot_res:
            for i in range(NUM_TIMES):
                depth = res["depth"]
                training_uuid = str(i)
                task = get_ssm_training_with_target_task(
                    training_dict=res,
                    lag=lag,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    seq_len=seq_len,
                    progress_bar_actor=progress_bar_actor,
                    save_dir=save_dir,
                    device=device,
                    training_uuid=training_uuid,
                    log_param_every=log_param_freq,
                    depth=depth,
                    target=target
                )
                if task is not None:
                    tasks.append(task)
    return tasks


def run_exp():
    ray.init(num_cpus=16, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    tasks = []
    tasks += get_tasks_induction_head(progress_bar_actor, 
                                      log_param_freq=log_param_every)
    tasks += get_tasks_fssm(progress_bar_actor,
                            log_param_freq=log_param_every)

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    run_exp()
