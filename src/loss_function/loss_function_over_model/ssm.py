import torch


# TODO: Write test for this
def l2_loss_with_random_noise(model: torch.nn.Module,
                              lag: int,
                              seq_len: int) -> torch.Tensor:
    k = model.get_kernel(seq_len)
    # v of ind rv is sum of rv variance
    # v of a*r where r is normal, is a**2
    variance = torch.sum(torch.abs(k[:lag]) ** 2)
    variance += torch.sum(k[lag + 1:] ** 2)
    variance += torch.abs(1 - k[lag]) ** 2
    loss = variance
    return loss
