import torch
import os
import random
import numpy as np

from src.experiments.utils.misc import opt_str_to_opt
from src.training.train_ssm import train_smm_over_white_noise_lag_multiprocess
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.models.factories.classical_ssm_factories import get_flexible_diag_ssm

import ray
from itertools import product

from src.multiprocessing.utils import ProgressBar


def init_diag_real(n, input_dim, output_dim):
    raise


def init_diag_complex(training_dict):
    def P2R(radii, angles):
        return radii * torch.exp(torch.tensor(1j * angles))

    def get_uniform_radii_in_ring(a, b):
        return random.uniform(a ** 2, b ** 2) ** 0.5

    def sample_num_from_dist_tuple(dist_tuple):
        # Given dist_tuple a list of tuple each consists of a range and a weight,
        # we want to sample a range with the respected probability of it's weight divided by the total weight
        # After we sample a range, we sample from that range uniformly
        list_of_range_canidates = [dist_tuple[i][0] for i in range(len(dist_tuple))]
        list_of_range_weights = np.array([dist_tuple[i][1] for i in range(len(dist_tuple))])
        list_of_range_weights = list_of_range_weights / sum(list_of_range_weights)
        chosen_interval_idx = np.random.choice(len(list_of_range_canidates), 1, p=list_of_range_weights)[0]
        chosen_interval = list_of_range_canidates[chosen_interval_idx]
        chosen_number = np.random.uniform(chosen_interval[0], chosen_interval[1])
        return chosen_number

    radii_range = training_dict['radii_range']
    B_std_real = training_dict['B_std_real']
    B_std_complex = training_dict['B_std_complex']
    angles_dist_tuple = training_dict['angles_dist_tuple']

    def init_func(n, input_dim, output_dim):
        A = []
        for i in range(n):
            radii = get_uniform_radii_in_ring(radii_range[0], radii_range[1])
            angle = sample_num_from_dist_tuple(angles_dist_tuple)
            A.append(P2R(radii, angle))
        A = torch.tensor(A)
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
    is_complex=training_dict["is_complex"]
    training_param = training_dict["training_param"]
    seq_len = training_dict["seq_len"]
    lag = training_dict["lag"]
    hidden_size = training_dict["hidden_size"]
    epochs = training_dict["epochs"]

    if is_complex:
        init = init_diag_complex(training_dict)

    else:
        init = init_diag_real(training_dict)

    model = get_flexible_diag_ssm(
        init_func=init,
        num_hidden_state=hidden_size,
        trainable_param_list=training_param,
        device=device,
        is_complex=is_complex,
    )

    logger = SSMTrainingLogger(saving_path=os.path.join(save_dir,training_uuid),
                               running_params=training_dict,
                               param_storing_freq=log_param_every,
                               saving_freq=10000)
    task = train_smm_over_white_noise_lag_multiprocess.remote(
        model=model,
        lag=lag,
        seq_len=seq_len,
        logger=logger,
        num_epochs=epochs,
        optimizer=opt,
        lr=lr,
        early_stop=False,
        min_cut=epochs,
        plot=False,
        progress_bar_actor=progress_bar_actor
    )
    return task


MAX_TASKS_NUM_AT_QUEUE = 64

def exp():
    exp_name = "limiting_to_distribution"

    ray.init(num_cpus=32, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    seq_lens = [32, 64]
    epochs = 100000

    cur_task_num = 1
    tasks = []

    hidden_sizes = [1048, 2048, 4096, 8192]
    radii_ranges = [
        [0, 1],
        [0, 0.95],
        [0.9, 0.9],
        [0.95, 0.95],
        [0.99, 0.99],
    ]
    #lrs = [0.00001, 0.0001, 0.001]
    lrs = [0.00001]

    def get_angle_dist_tuple(until_what_percent_of_angles, prop_to_rest):
        if until_what_percent_of_angles != 1:
            return [
                [
                    [-np.pi * until_what_percent_of_angles, np.pi * until_what_percent_of_angles],
                    prop_to_rest
                ],
                [
                    [np.pi * until_what_percent_of_angles, np.pi * (2 - until_what_percent_of_angles)],
                    1
                ],
            ]
        else:
            return [
                [
                    [0, 2*np.pi],
                    1
                ]
            ]

    base_mult = 3
    angles_dist_tuples = [
        get_angle_dist_tuple(1 / 8, base_mult),
        get_angle_dist_tuple(1 / 4, base_mult),
        get_angle_dist_tuple(1 / 2, base_mult),
        get_angle_dist_tuple(1 / 4, base_mult * 2),
        get_angle_dist_tuple(1 / 2, base_mult * 4),
        get_angle_dist_tuple(1, 1),
    ]
    base_mult = 9
    angles_dist_tuples += [
        get_angle_dist_tuple(1 / 8, base_mult),
        get_angle_dist_tuple(1 / 4, base_mult),
        get_angle_dist_tuple(1 / 2, base_mult),
        get_angle_dist_tuple(1 / 4, base_mult * 2),
        get_angle_dist_tuple(1 / 2, base_mult * 4),
        get_angle_dist_tuple(1, 1),
    ]

    #B_stds = [0.0001, 1]
    B_stds = [0.0001]
    for seq_len in seq_lens:
        lag = seq_len - 3
        for B_std in B_stds:
            for lr in lrs:
                for train_param in [["A", "B"], ["B"]]:
                    for radii_range in radii_ranges:
                        for angles_dist_tuple in angles_dist_tuples:
                            for hidden_size in hidden_sizes:
                                # if cur_task_num > 600:
                                #     cur_task_num += 1
                                #     continue
                                # if cur_task_num < 480:
                                #     cur_task_num += 1
                                #     continue
                                if cur_task_num > 480:
                                    cur_task_num += 1
                                    continue
                                tasks.append(
                                    get_ssm_training_task(
                                        training_dict={
                                            'lr': lr,
                                            'opt': 'SGD',
                                            'model_name': 'diag_complex',
                                            'B_std_real': B_std,
                                            'B_std_complex': B_std,
                                            'hidden_size': hidden_size,
                                            'radii_range': radii_range,
                                            'angles_dist_tuple': angles_dist_tuple,
                                            'lag': lag,
                                            'epochs': epochs,
                                            'seq_len': seq_len,
                                            'training_param': train_param,
                                            'is_complex': True
                                        },

                                        save_dir=os.path.join("..", "results", "LDSSM", "3limiting_to_dist"),
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
