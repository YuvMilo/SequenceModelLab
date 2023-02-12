from collections import defaultdict
import torch


class BaseTrainingLogger:

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, loss: float, epoch_num: int, model: torch.nn.Module) -> None:
        self.history["loss"].append(loss)

    def save(self) -> None:
        pass

    def __enter__(self) -> None:
        return self

    def __del__(self):
        self.save()
