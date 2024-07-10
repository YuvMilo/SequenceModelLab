import os
import torch
import ray
import numpy as np

from src.multiprocessing.utils import ProgressBar
from src.experiments.utils.best_settings import top_rot_res_linear
from src.experiments.utils.get_standard_tasks_linear_ssm import get_rot_training_task

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_param_every = 8
save_dir = os.path.join("..", "results", "rot_abo")
hidden_sizes = [128]
seq_len = np.max(hidden_sizes) * 2
epochs = 20000

NUM_TIMES = 5

main_diagonal_diffs = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 4]
off_diagonal_ratios = [1, 1.2, 1.5, 1.7, 2, 4, 8, 16, 32, 64, 128]


def get_tasks_rot(progress_bar_actor,
                  log_param_freq: int = 1):
    tasks = []
    for hidden_size in hidden_sizes:
        lag = hidden_size - 2
        for i in range(NUM_TIMES):
            for res in top_rot_res_linear:
                for off_diagonal_ratio in off_diagonal_ratios:
                    for main_diagonal_diff in main_diagonal_diffs:
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
                            log_param_every=log_param_freq,
                            off_diagonal_ratio=off_diagonal_ratio,
                            main_diagonal_diff=main_diagonal_diff
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

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    run_exp()
