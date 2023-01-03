import torch


def delay_l2(lag):
    def delay_loss_func(output, target):
        loss = torch.mean((output[:, lag:, :] - target[:, lag:, :])**2)
        return loss
    return delay_loss_func
