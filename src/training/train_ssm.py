import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ray

from src.loss_function.loss_function_over_model.ssm import l2_loss_with_random_noise
from src.loss_function.loss_function_over_model.ssm import l2_loss_with_target
from src.models.ssm import SMMModel
from src.logging.training_logger.ssm_logger import SSMTrainingLogger
from src.data_loaders.data_sets.delay_data_set import DelayedSignalDatasetRegenerated
from src.data_loaders.data_sets.utils.signals import NormalNoiseSignalGenerator
from src.loss_function.loss_function_over_data.delay import delay_l2
from src.multiprocessing.utils import ProgressBarActor


def train_smm_over_white_noise_lag(
        model: SMMModel, lag: int,
        seq_len: int,
        optimizer: torch.optim.Optimizer,
        lr: float,
        num_epochs: int, logger: SSMTrainingLogger,
        plot: bool = True,
        min_cut: int = 1000,
        early_stop: bool = False,
        show_tqdm: bool = True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True, disable=not show_tqdm):
        # Clear the gradients
        optimizer.zero_grad()

        loss = l2_loss_with_random_noise(model, lag, seq_len)

        if np.isnan(loss.item()):
            logger.save()
            break

        # Backward pass
        loss.backward()

        logger.log(loss=loss.item(),
                   epoch_num=epoch,
                   model=model)

        optimizer.step()

        # TODO - make this a debugging/plotting strategy for decoupling
        if plot and (epoch %500 == 0):
            plt.close("all")

            figure, axis = plt.subplots(4, 1)
            axis[0].plot(model.get_kernel(seq_len*2).cpu().detach().numpy(), color="orange")
            ker = model.get_kernel(1000).cpu().detach().numpy()
            axis[1].plot(ker, color="orange")
            fft = np.fft.fft(ker).real
            axis[2].plot(np.linspace(0,2*np.pi,len(fft)+1)[:-1],fft)
            axis[3].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10**6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.3):
            no_change_time = 200
            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if pre_min_loss - current_min_loss < 0.01:
                    logger.save()
                    break
        if all_loss[-1] < 0.03:
            logger.save()
            break

    logger.save()


@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
def train_smm_over_white_noise_lag_multiprocess(
        model: SMMModel, lag: int,
        seq_len: int,
        optimizer: torch.optim.Optimizer,
        lr: float,
        num_epochs: int, logger: SSMTrainingLogger,
        plot: bool = False,
        min_cut: int = 1000,
        early_stop: bool = False,
        show_tqdm: bool = False,
        progress_bar_actor: ProgressBarActor = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True, disable=not show_tqdm):
        # Clear the gradients
        optimizer.zero_grad()

        loss = l2_loss_with_random_noise(model, lag, seq_len)

        if np.isnan(loss.item()):
            logger.save()
            break

        # Backward pass
        loss.backward()

        logger.log(loss=loss.item(),
                   epoch_num=epoch,
                   model=model)

        optimizer.step()

        # TODO - make this a debugging/plotting strategy for decoupling
        if plot and (epoch % 20 == 0):
            plt.close("all")

            figure, axis = plt.subplots(2, 1)
            axis[0].plot(model.get_kernel(1000).cpu().detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10**6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.3):
            no_change_time = 200
            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if pre_min_loss - current_min_loss < 0.01:
                    logger.save()
                    break

        if all_loss[-1] < 0.001:
            logger.save()
            break

    if progress_bar_actor is not None:
        progress_bar_actor.update.remote()


    logger.save()



@ray.remote(num_gpus=0.25 if torch.cuda.is_available() else 0)
def train_smm_over_white_noise_with_target_multiprocess(
        model: SMMModel, target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        lr: float,
        num_epochs: int, logger: SSMTrainingLogger,
        plot: bool = False,
        min_cut: int = 10000,
        early_stop: bool = False,
        show_tqdm: bool = False,
        progress_bar_actor: ProgressBarActor = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    target = target.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True, disable=not show_tqdm):
        # Clear the gradients
        optimizer.zero_grad()

        loss = l2_loss_with_target(model, target)

        if np.isnan(loss.item()):
            logger.save()
            break

        # Backward pass
        loss.backward()

        logger.log(loss=loss.item(),
                   epoch_num=epoch,
                   model=model)

        optimizer.step()

        # TODO - make this a debugging/plotting strategy for decoupling
        if plot and (epoch % 20 == 0):
            plt.close("all")

            figure, axis = plt.subplots(2, 1)
            axis[0].plot(model.get_kernel(1000).cpu().detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10**6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.01):
            no_change_time = 9900
            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if ((pre_min_loss - current_min_loss)/current_min_loss < (0.1/100)) or (pre_min_loss - current_min_loss < 0.001):
                    logger.save()
                    break

        if all_loss[-1] < 0.0005 and len(all_loss) > min_cut:
            logger.save()
            break

    if progress_bar_actor is not None:
        progress_bar_actor.update.remote()


    logger.save()



def train_ssm(model: SMMModel,
              lag: int,
              seq_len: int,
              samples_num: int,
              optimizer: torch.optim.Optimizer,
              lr: float,
              num_epochs: int, logger: SSMTrainingLogger,
              plot: bool = True,
              min_cut: int = 1000,
              early_stop: bool = False,
              show_tqdm: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    criterion = delay_l2(lag)
    signal_generator = NormalNoiseSignalGenerator()
    ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
                                         seq_length=seq_len,
                                         signal_generator=signal_generator)
    dl = DataLoader(ds, batch_size=samples_num)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True, disable=not show_tqdm):
        epoch_loss = 0
        # Loop over the data in the dataloader
        for input_data, labels in dl:
            input_data = input_data.to(device)
            labels = labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_data)
            loss = criterion(output, labels)

            if np.isnan(loss.item()):
                logger.save()
                break

            epoch_loss += loss.cpu().item()

            # Backward pass
            loss.backward()
            optimizer.step()

        logger.log(loss=epoch_loss / len(dl),
                   epoch_num=epoch,
                   model=model)

        # TODO - make this a debugging/plotting strategy for decoupling
        if plot and (epoch % 20 == 0):
            plt.close("all")

            figure, axis = plt.subplots(2, 1)
            axis[0].plot(model.get_kernel(1000).cpu().detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10 ** 6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.3):
            no_change_time = 200
            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if pre_min_loss - current_min_loss < 0.01:
                    logger.save()
                    break

    logger.save()


@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
def train_ssm_multiprocess(model: SMMModel,
                           lag: int,
                           seq_len: int,
                           samples_num: int,
                           optimizer: torch.optim.Optimizer,
                           lr: float,
                           num_epochs: int, logger: SSMTrainingLogger,
                           plot: bool = False,
                           min_cut: int = 1000,
                           early_stop: bool = False,
                           show_tqdm: bool = False,
                           progress_bar_actor: ProgressBarActor = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)
    criterion = delay_l2(lag)
    signal_generator = NormalNoiseSignalGenerator()
    ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=samples_num,
                                         seq_length=seq_len,
                                         signal_generator=signal_generator)
    dl = DataLoader(ds, batch_size=samples_num)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True, disable=not show_tqdm):
        epoch_loss = 0
        # Loop over the data in the dataloader
        for input_data, labels in dl:
            input_data = input_data.to(device)
            labels = labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_data)
            loss = criterion(output, labels)

            if np.isnan(loss.item()):
                logger.save()
                break

            epoch_loss += loss.cpu().item()

            # Backward pass
            loss.backward()
            optimizer.step()

        logger.log(loss=epoch_loss / len(dl),
                   epoch_num=epoch,
                   model=model)

        # TODO - make this a debugging/plotting strategy for decoupling
        if plot and (epoch % 20 == 0):
            plt.close("all")

            figure, axis = plt.subplots(2, 1)
            axis[0].plot(model.get_kernel(200).cpu().detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10 ** 6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.01):
            no_change_time = 9900
            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if ((pre_min_loss - current_min_loss) / current_min_loss < (0.1 / 100)) or (
                        pre_min_loss - current_min_loss < 0.001):
                    logger.save()
                    break

    if progress_bar_actor is not None:
        progress_bar_actor.update.remote()

    logger.save()
