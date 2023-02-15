import torch


opt_str_to_opt = {
    "adam": torch.optim.Adam,
    "SGD": torch.optim.SGD
}

opt_to_opt_str = {
    torch.optim.Adam: "adam",
    torch.optim.SGD: "SGD"
}
