import torch
import os
import random

from src.experiments.utils.misc import opt_str_to_opt
from src.training.train_ssm import train_smm_over_white_noise_lag_multiprocess
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.models.factories.classical_ssm_factories import get_flexible_diag_ssm

import ray
from itertools import product

from src.multiprocessing.utils import ProgressBar

def get_ssm_training_task(training_dict: dict,
                          lag: int, hidden_size: int, epochs: int,
                          seq_len: int,
                          save_dir: str,
                          device: torch.device,
                          is_complex: bool,
                          radii_range,
                          angle_range,
                          training_param,
                          progress_bar_actor,
                          training_uuid: str = "",
                          log_param_every: int = 1,
                          ):
    training_dict["lag"] = lag
    training_dict["hidden_size"] = hidden_size
    training_dict["epochs"] = epochs
    training_dict["seq_len"] = seq_len
    training_dict["is_complex"] = is_complex
    training_dict["radii_range"] = str(radii_range)
    training_dict["angle_range"] = str(angle_range)
    training_dict["training_param"] = str(training_param)
    opt = opt_str_to_opt[training_dict["opt"]]
    lr = float(training_dict["lr"])
    B_std_real = float(training_dict["B_std_real"])
    B_std_complex = float(training_dict["B_std_real"])

    def init_diag_real(n, input_dim, output_dim):
        A = torch.linspace(radii_range[0], radii_range[1], n//2+2)[1:-1]
        A = torch.tensor(list(A)+list(-1*A))
        B = B_std_real * torch.randn([n, input_dim], dtype=torch.float)
        D = torch.zeros(output_dim)
        C = torch.ones([output_dim, n], dtype=torch.float)
        return A, B, C, D

    # def init_diag_complex(n, input_dim, output_dim):
    #     A = torch.linspace(0, 1, n//2+1)[:-1]
    #     A = torch.tensor((list(A)+list(-A)))
    #     A = torch.complex(A, torch.zeros_like(A))
    #     A += complex_noise*torch.randn(A.shape) * 1j
    #     B = B_std * torch.randn([n, input_dim])
    #     B = torch.complex(B, torch.zeros_like(B))
    #     B += complex_noise * torch.randn(B.shape) * 1j
    #     D = torch.zeros(output_dim)
    #     D = torch.complex(D, torch.zeros_like(D))
    #     C = torch.ones([output_dim, n])
    #     C = torch.complex(C, torch.zeros_like(C))
    #     return A, B, C, D

    def init_diag_complex(n, input_dim, output_dim):
        def P2R(radii, angles):
            return radii * torch.exp(torch.tensor(1j * angles))
            # A = torch.linspace(0, 1, n//2+1)[:-1]
        # A = torch.tensor((list(A)+list(-A)))
        # A = torch.complex(A, torch.zeros_like(A))
        # A += complex_noise*torch.randn(A.shape) * 1j
        #angles = torch.linspace(0,2*torch.pi,n+1)[:-1]
        #A = [P2R(radii,ang) for ang in angles]
        A = []
        for i in range(n):
            radii = random.uniform(radii_range[0], radii_range[1])
            sign = random.choice([-1, 1])
            angle = random.uniform(angle_range[0], angle_range[1])
            A.append(sign*P2R(radii, angle))
        A = torch.tensor(A)
        B = B_std_real * torch.randn([n, input_dim]) + 1j * B_std_complex * torch.randn([n, input_dim])
        D = torch.zeros(output_dim)
        D = torch.complex(D, torch.zeros_like(D))
        C = torch.ones([output_dim, n])
        C = torch.complex(C, torch.zeros_like(C))
        return A, B, C, D

    if is_complex == True:
        init = init_diag_complex

    else:
        init = init_diag_real



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


def complex_exp():
    ray.init(num_cpus=16, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    lag = 26
    seq_len = 32
    hidden_size = 64
    epochs = 100000

    tasks = []
    for hidden_size in [64,256]:
        for angle_range in [
            [-torch.pi / 8, torch.pi / 8],
            [-torch.pi / 4, torch.pi / 4],
            [-torch.pi / 2, torch.pi / 2]
        ]:
            angle_range_str = "_ar_"+str(torch.pi/angle_range[1])
            for radii_range in [[0, 1], [0.9, 1], [0.5, 0.6]]:
                for lr in [0.00001, 0.0001, 0.001, 0.01]:
                    for train_param in [["B"], ["A", "B"]]:
                        if "A" in train_param:
                            tp_str = "A_and_B"
                        else:
                            tp_str = "Only_B"
                        tasks.append(
                            get_ssm_training_task(
                                training_dict={
                                    'lr': str(lr),
                                    'opt': 'SGD',
                                    'model_name': 'diag_complex',
                                    'B_std_real': '0.0001',
                                    'B_std_complex': '0.0001'
                                },
                                lag=lag,
                                hidden_size=hidden_size,
                                epochs=epochs,
                                seq_len=seq_len,
                                save_dir=os.path.join("..", "results", "LDSSM", "complex"),
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                training_uuid="l_" + str(lr) + "_r_" + str(radii_range) + "_tp_" + tp_str+angle_range_str+"h_"+str(hidden_size),
                                log_param_every=30,
                                is_complex=True,
                                radii_range=radii_range,
                                angle_range=angle_range,
                                training_param=train_param,
                                progress_bar_actor=progress_bar_actor
                            )
                        )

    pb.set_total(len(tasks))
    pb.print_until_done()

if __name__ == "__main__":
    real_exp()
