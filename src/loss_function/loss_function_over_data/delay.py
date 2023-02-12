import torch


# TODO: Write test for this
def delay_l2(lag):
    def delay_loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((output[:, lag:, :] - target[:, lag:, :])**2)
        return loss
    return delay_loss_func
