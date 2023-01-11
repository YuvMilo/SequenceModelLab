import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def l2_loss_with_random_noise(model, lag, seq_len):
    k = model.get_kernel(seq_len)
    # v of ind rv is sum of rv variance
    # v of a*r where r is normal, is a**2
    variance = torch.sum(torch.abs(k[:lag]) ** 2)
    variance += torch.sum(k[lag+1:] ** 2)
    variance += torch.abs(1 - k[lag]) ** 2
    loss = variance
    return loss


def train(model, dl, criterion, optimizer, num_epochs, logger, plot=True):

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True):
        epoch_loss = 0
        # Loop over the data in the dataloader
        for input_data, labels in dl:
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_data)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
        logger.log(loss=epoch_loss / len(dl),
                   epoch_num=epoch,
                   data_loader=dl,
                   model=model)

        if plot and (epoch % 10 == 0):
            plt.close("all")

            # TODO - make this a debugging strategy for decoupling
            figure, axis = plt.subplots(3, 1)
            axis[0].plot(output[0, :100].detach().numpy(), color="orange")
            axis[0].plot(labels[0, :100].detach().numpy(), color="red")
            axis[0].plot(input_data[0, :100].detach().numpy(), color="blue")

            axis[1].plot(model.get_kernel(1000).detach().numpy(), color="orange")

            axis[2].plot([e.entity for e in logger.history["loss"][-150:]])
            plt.show()
            pass


def train_smm_random_noise_fast(model, lag, seq_len, optimizer, num_epochs, logger,
                               plot=False,
                               min_cut=1000):

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs), leave=True):
        # Clear the gradients
        optimizer.zero_grad()

        loss = l2_loss_with_random_noise(model, lag, seq_len)

        # Backward pass
        loss.backward()
        optimizer.step()
        logger.log(loss=loss.item(),
                   epoch_num=epoch,
                   data_loader=None,
                   model=model)

        if plot and (epoch % 50 == 0):
            plt.close("all")

            # TODO - make this a debugging strategy for decoupling
            figure, axis = plt.subplots(2, 1)
            axis[0].plot(model.get_kernel(1000).detach().numpy(), color="orange")

            axis[1].plot(logger.get_loss_hist()[-200:])
            plt.show()
            pass

        # TODO this should be a early stopping strategy
        all_loss = np.array(logger.get_loss_hist())

        if all_loss[-1] > 100:
            logger.save()
            break

        if (len(all_loss) > min_cut) or (np.min(all_loss) < 0.3):
            arg_min_loss = np.argmin(all_loss) + 1
            ago = len(all_loss) - arg_min_loss
            if ago > 200:
                logger.save()
                break

            no_change_time = 200
            if len(all_loss) > no_change_time:
                current_min_loss = np.min(all_loss[-no_change_time:])
                pre_min_loss = np.min(all_loss[:-no_change_time])
                if pre_min_loss-current_min_loss < 0.01:
                    logger.save()
                    break
