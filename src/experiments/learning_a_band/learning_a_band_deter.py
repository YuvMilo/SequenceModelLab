import torch
import os
import random
import numpy as np

from src.experiments.utils.misc import opt_str_to_opt
from src.training.train_ssm import train_smm_over_white_noise_with_target_multiprocess,train_smm_over_white_noise_lag
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.models.factories.classical_ssm_factories import get_band_ssm

import ray
from itertools import product

from src.multiprocessing.utils import ProgressBar


def band_init(training_dict):
    B_std_real = training_dict['B_std_real']
    B_std_complex = training_dict['B_std_complex']
    band_angles_init = training_dict['band_angles_init']

    def init_func(n, input_dim, output_dim):
        A = torch.tensor([band_angles_init])
        B = B_std_real * torch.randn([n, input_dim]) + 1j * B_std_complex * torch.randn([n, input_dim])
        D = torch.zeros(output_dim)
        D = torch.complex(D, torch.zeros_like(D))
        C = torch.ones([output_dim, n])
        C = torch.complex(C, torch.zeros_like(C))
        return A, B, C, D

    return init_func


def get_ssm_training_task(training_dict: dict,
                          save_dir: str,
                          device: torch.device,
                          progress_bar_actor,
                          training_uuid: str = "",
                          log_param_every: int = 1,
                          ):
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = training_dict["lr"]
    training_param = training_dict["training_param"]
    seq_len = training_dict["seq_len"]
    hidden_size = training_dict["hidden_size"]
    epochs = training_dict["epochs"]
    target_freq = training_dict["target_freq"]

    init = band_init(training_dict)

    model = get_band_ssm(
        init_func=init,
        num_hidden_state=hidden_size,
        trainable_param_list=training_param,
        device=device
    )

    logger = SSMTrainingLogger(saving_path=os.path.join(save_dir,training_uuid),
                               running_params=training_dict,
                               param_storing_freq=log_param_every,
                               saving_freq=10000)

    def get_target_in_freq(seq_len, freq=1 / 16, noise_std=0):
        a = torch.zeros(seq_len, dtype=torch.complex64)
        freq_idx_cutoff = int(len(a) * freq / 2)
        a[0] = 1
        a[1:freq_idx_cutoff + 1] += 1 + 1j
        a[-1 * freq_idx_cutoff:] += 1 - 1j

        f = torch.fft.ifft(a).real

        # Normalize the tensor and scale it to have a total power of 1
        f /= torch.sqrt(torch.sum(f ** 2) / f.numel())

        f += torch.randn(f.shape) * noise_std
        return f

    target = get_target_in_freq(seq_len, freq=target_freq)

    task = train_smm_over_white_noise_with_target_multiprocess.remote(
        model=model,
        target=target,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=True,
        min_cut=epochs,
        plot=False,
        progress_bar_actor=progress_bar_actor
    )
    return task
    # train_smm_over_white_noise_lag(
    #     model=model,
    #     logger=logger,
    #     num_epochs=epochs,
    #     optimizer=opt,
    #     lr=lr,
    #     early_stop=False,
    #     min_cut=epochs,
    #     plot=True,
    #     lag=seq_len-5,
    #     seq_len=seq_len
    # )


MAX_TASKS_NUM_AT_QUEUE = 64

def exp():
    exp_name = "deter"

    ray.init(num_cpus=32, ignore_reinit_error=True)

    pb = ProgressBar()
    progress_bar_actor = pb.actor

    seq_lens = [32, 48, 64]
    epochs = 100000

    cur_task_num = 1
    tasks = []

    hidden_sizes = [256]

    lrs = [0.0001, 0.00005]

    band_angles_inits = [0.5, 0.25, 0.1]
    band_angles_inits = [b*np.pi for b in band_angles_inits]

    freqs = [1, 0.7, 0.5, 0.25]

    B_stds = [0.00005]

    for B_std in B_stds:
        for opt in ["SGD", "adam"]:
            for train_param in [["A", "B"]]:
                for lr in lrs:
                    for seq_len in seq_lens:
                        for target_freq in freqs:
                            for band_angles_init in band_angles_inits:
                                for hidden_size in hidden_sizes:
                                    tasks.append(
                                        get_ssm_training_task(
                                            training_dict={
                                                'lr': lr,
                                                'opt': opt,
                                                'model_name': 'diag_complex_band',
                                                'B_std_real': B_std,
                                                'B_std_complex': B_std,
                                                'hidden_size': hidden_size,
                                                'epochs': epochs,
                                                'seq_len': seq_len,
                                                'training_param': train_param,
                                                'is_complex': True,
                                                "target_freq": target_freq,
                                                "band_angles_init": band_angles_init
                                            },

                                            save_dir=os.path.join("..", "results", "band_learning", exp_name),
                                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                            training_uuid=exp_name + "_" + str(cur_task_num).zfill(4),
                                            log_param_every=30,
                                            progress_bar_actor=progress_bar_actor
                                        )
                                    )
                                    print(cur_task_num)
                                    cur_task_num += 1
                                # if cur_task_num % MAX_TASKS_NUM_AT_QUEUE == 0:
                                #     pb.set_total(len(tasks))
                                #     pb.print_until_done()
                                #     pb = ProgressBar()
                                #     progress_bar_actor = pb.actor
                                #     tasks = []



    pb.set_total(len(tasks))
    pb.print_until_done()

if __name__ == "__main__":
    exp()
