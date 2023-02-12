import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ray

from src.loss_function.loss_function_over_model.ssm import l2_loss_with_random_noise
from src.models.ssm import SMMModel
from src.logging.training_logger import ssm_logger

from src.multiprocessing.utils import ProgressBarActor


def train_smm_over_white_noise_lag(model: SMMModel, lag: int,
                                   seq_len: int, optimizer: torch.optim.Optimizer,
                                   num_epochs: int, logger: ssm_logger,
                                   device: torch.device = None,
                                   plot: bool = False,
                                   min_cut: int = 1000,
                                   early_stop: bool = True,
                                   show_tqdm: bool = True):

    if device is not None:
        model.change_device(device)

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
            axis[0].plot(model.get_kernel(1000).detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 10**6:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.3):
            arg_min_loss = np.argmin(all_loss) + 1
            ago = len(all_loss) - arg_min_loss
            if early_stop and ago > 200:
                logger.save()
                break

            # if ago > 2000:
            #     logger.save()
            #     break
            no_change_time = 200
            # if early_stop:
            #     no_change_time = 200
            # else:
            #     no_change_time = 2000

            if early_stop and len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if pre_min_loss - current_min_loss < 0.01:
                    logger.save()
                    break

    logger.save()


@ray.remote(num_gpus=0.25)
def train_smm_over_white_noise_lag_multiprocess(
        model: SMMModel, lag: int,
        seq_len: int,
        optimizer: torch.optim.Optimizer,
        num_epochs: int, logger: ssm_logger,
        device: torch.device = None,
        plot: bool = False,
        min_cut: int = 1000,
        early_stop: bool = True,
        show_tqdm: bool = False,
        progress_bar_actor: ProgressBarActor = None):

    if device is not None:
        model.change_device(device)

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
            axis[0].plot(model.get_kernel(1000).detach().numpy(), color="orange")

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

    if progress_bar_actor is not None:
        progress_bar_actor.update.remote()

    logger.save()

# def train(model, dl, criterion, optimizer, num_epochs, logger, plot=True):
#
#     # Loop over the epochs
#     for epoch in tqdm(range(num_epochs), leave=True):
#         epoch_loss = 0
#         # Loop over the data in the dataloader
#         for input_data, labels in dl:
#             # Clear the gradients
#             optimizer.zero_grad()
#
#             # Forward pass
#             output = model(input_data)
#             loss = criterion(output, labels)
#             epoch_loss += loss.cpu().item()
#
#             # Backward pass
#             loss.backward()
#             optimizer.step()
#
#         logger.log(loss=epoch_loss / len(dl),
#                    epoch_num=epoch,
#                    data_loader=dl,
#                    model=model)
#
#         # TODO - make this a debugging/plotting strategy for decoupling
#         if plot and (epoch % 10 == 0):
#             plt.close("all")
#
#             figure, axis = plt.subplots(3, 1)
#             axis[0].plot(output[0, :100].detach().numpy(), color="orange")
#             axis[0].plot(labels[0, :100].detach().numpy(), color="red")
#             axis[0].plot(input_data[0, :100].detach().numpy(), color="blue")
#
#             axis[1].plot(model.get_kernel(100).detach().numpy(), color="orange")
#
#             axis[2].plot([e.entity for e in logger.history["loss"][-150:]])
#             plt.show()
#             pass
