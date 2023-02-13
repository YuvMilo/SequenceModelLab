import os
import torch
import ray

from src.models.factories.classical_ssm_factories import get_rot_ssm_one_over_n, \
    get_full_ssm_diag_plus_noise1D, get_full_ssm_hippo1D, \
    get_rot_ssm_equally_spaced
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.training.train_ssm import train_smm_over_white_noise_lag_multiprocess
from src.multiprocessing.utils import ProgressBar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
lag = 63
seq_len = hidden_size * 3
epochs = 30000

NUM_TIMES = 20

opt_str_to_opt = {
    'adam': torch.optim.Adam,
    "SGD": torch.optim.SGD
}

top_fssm_res = [
    {'hidden': '64', 'lag': '60', 'lr': '0.001', 'noise': '0.001',
     'diag_init': '0.93', 'BC_std': '0.1', 'opt': 'adam',
     'model_name': 'ssm'},
    {'hidden': '64', 'lag': '60', 'lr': '0.001', 'noise': '0.01',
     'diag_init': '0.9', 'BC_std': '0.1', 'opt': 'adam',
     'model_name': 'ssm'},
    {'hidden': '64', 'lag': '60', 'lr': '0.01', 'noise': '0.01',
     'diag_init': '0.9', 'BC_std': '0.1', 'opt': 'SGD',
     'model_name': 'ssm'},
    {'hidden': '64', 'lag': '60', 'lr': '0.01', 'noise': '0.01',
     'diag_init': '0.93', 'BC_std': '0.001', 'opt': 'SGD',
     'model_name': 'ssm'},
]


def var_fssm(progress_bar_actor):
    def get_saving_path_for_exp(optimizer, lag, lr, noise, diags_init, BC_std, i):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplementedError()

        return os.path.join("..", "results", "variance_mult",
                            "ssm_h{hidden}_i{i}_l{lag}_lr{lr}_n{noise}_d{diag_init}_b{BC_std}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                noise=noise,
                                diag_init=diags_init,
                                BC_std=BC_std,
                                opt=opt,
                                i=i,
                                hidden=hidden_size
                            ))

    tasks = []
    for res in top_fssm_res:
        opt = opt_str_to_opt[res["opt"]]
        lr = float(res["lr"])
        noise = float(res["noise"])
        diags_init = float(res["diag_init"])
        BC_std = float(res["BC_std"])
        for i in range(NUM_TIMES):
            model = get_full_ssm_diag_plus_noise1D(
                num_hidden_state=hidden_size,
                device=device,
                B_init_std=BC_std,
                C_init_std=BC_std,
                A_noise_std=noise,
                A_diag=diags_init
            )

            saving_path = get_saving_path_for_exp(opt, lag, lr,
                                                  noise, diags_init, BC_std,
                                                  i=i)

            logger = SSMTrainingLogger(saving_path=saving_path,
                                       running_params=res)
            tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=opt,
                lr=lr,
                early_stop=False,
                min_cut=epochs,
                progress_bar_actor=progress_bar_actor
            ))
    return tasks


top_hippo_res = [
    {'hidden': '64', 'lag': '60', 'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo'},
    {'hidden': '64', 'lag': '60', 'lr': '0.01', 'dt': '0.05',
     'opt': 'adam', 'model_name': 'hippo'},
    {'hidden': '64', 'lag': '60', 'lr': '0.0001', 'dt': '0.05',
     'opt': 'SGD', 'model_name': 'hippo'},
    {'hidden': '64', 'lag': '60', 'lr': '5e-05', 'dt': '0.1',
     'opt': 'SGD', 'model_name': 'hippo'}
]


def var_hippo(progress_bar_actor):
    def get_saving_path_for_exp(optimizer, lag, lr, dt, i):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplementedError()

        return os.path.join("..", "results", "variance_mult",
                            "hippo_h{hidden}_i{i}_l{lag}_lr{lr}_dt{dt}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                dt=dt,
                                i=i,
                                hidden=hidden_size
                            ))

    tasks = []
    for res in top_hippo_res:
        opt = opt_str_to_opt[res["opt"]]
        lr = float(res["lr"])
        dt = float(res["dt"])

        for i in range(NUM_TIMES):
            model = get_full_ssm_hippo1D(num_hidden_state=hidden_size,
                                         device=device,
                                         dt=dt)

            saving_path = get_saving_path_for_exp(opt, lag, lr, dt, i)

            logger = SSMTrainingLogger(saving_path=saving_path,
                                       running_params=res)
            tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=opt,
                lr=lr,
                early_stop=False,
                min_cut=epochs,
                progress_bar_actor=progress_bar_actor
            ))
    return tasks


top_hippo_no_res = [
    {'no': 'a', 'hidden': '64', 'lag': '60', 'lr': '0.1', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_a'},
    {'no': 'ac', 'hidden': '64', 'lag': '60', 'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_ac'},
]


def var_hippo_no_a(progress_bar_actor):
    def get_saving_path_for_exp(optimizer, lag, lr, dt, no, i):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplementedError()

        return os.path.join("..", "results", "variance_mult",
                            "hippo_no{no}_h{hidden}_i{i}_l{lag}_lr{lr}_dt{dt}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                dt=dt,
                                no=no,
                                i=i,
                                hidden=hidden_size
                            ))

    tasks = []
    for res in top_hippo_no_res:
        opt = opt_str_to_opt[res["opt"]]
        lr = float(res["lr"])
        dt = float(res["dt"])
        no = res["no"]

        for i in range(NUM_TIMES):
            if no == "a":
                trainable_param_list = ["C", "B"]

            elif no == "ac":
                trainable_param_list = ["B"]
            else:
                raise NotImplementedError

            model = get_full_ssm_hippo1D(num_hidden_state=hidden_size,
                                         device=device,
                                         dt=dt,
                                         trainable_param_list=trainable_param_list)

            saving_path = get_saving_path_for_exp(opt, lag, lr, dt, no, i)

            logger = SSMTrainingLogger(saving_path=saving_path,
                                       running_params=res)
            tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=opt,
                lr=lr,
                early_stop=False,
                min_cut=epochs,
                progress_bar_actor=progress_bar_actor
            ))
    return tasks


top_rot_res = [
    {'rot_type': 'eq', 'hidden': '64', 'lag': '60', 'lr': '0.001',
     'opt': 'adam', 'model_name': 'rot'},
    {'rot_type': 'eq', 'hidden': '64', 'lag': '60', 'lr': '0.005',
     'opt': 'SGD', 'model_name': 'rot'}
]


def var_rot(progress_bar_actor):
    def get_saving_path_for_exp(optimizer, lag, lr, rot_type, i):
        if optimizer == torch.optim.SGD:
            opt = "SGD"
        elif optimizer == torch.optim.Adam:
            opt = "adam"
        else:
            raise NotImplementedError()

        return os.path.join("..", "results", "variance_mult",
                            "rot_h{hidden}_{rot_type}_i{i}_l{lag}_lr{lr}_o{opt}".format(
                                lag=lag,
                                lr=lr,
                                opt=opt,
                                rot_type=rot_type,
                                i=i,
                                hidden=hidden_size
                            ))

    tasks = []
    for res in top_rot_res:
        opt = opt_str_to_opt[res["opt"]]
        lr = float(res["lr"])
        rot_type = res["rot_type"]
        for i in range(NUM_TIMES):

            if rot_type == "one_over":
                model = get_rot_ssm_one_over_n(num_hidden_state=hidden_size,
                                               device=device)
            elif rot_type == "eq":
                model = get_rot_ssm_equally_spaced(num_hidden_state=hidden_size,
                                                   device=device)
            else:
                raise NotImplementedError()

            saving_path = get_saving_path_for_exp(opt, lag, lr, rot_type, i)

            logger = SSMTrainingLogger(saving_path=saving_path,
                                       running_params=res)
            tasks.append(train_smm_over_white_noise_lag_multiprocess.remote(
                model=model,
                lag=lag,
                seq_len=seq_len,
                logger=logger,
                num_epochs=epochs,
                optimizer=opt,
                lr=lr,
                early_stop=False,
                min_cut=epochs,
                progress_bar_actor=progress_bar_actor
            ))
    return tasks


def run_exp():
    ray.init()
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    tasks = []
    tasks += var_hippo_no_a(progress_bar_actor)
    tasks += var_rot(progress_bar_actor)
    tasks += var_hippo(progress_bar_actor)
    tasks += var_fssm(progress_bar_actor)

    pb.set_total(len(tasks))
    pb.print_until_done()


if __name__ == "__main__":
    run_exp()
