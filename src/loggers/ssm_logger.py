from dataclasses import dataclass
from collections import defaultdict
from typing import Any
import pickle
import os

from src.models.ssm import SMMModel

from src.loggers.base_logger import BaseLogger


@dataclass
class LoggingEntity:
    entity: Any
    epoch_logged: int


class SSMLogger(BaseLogger):

    def __init__(self, saving_path, kernel_saving_size=1024, saving_freq=500):
        self.saving_freq = saving_freq
        self.saving_path = saving_path
        self.kernel_saving_size = kernel_saving_size
        self.history = defaultdict(list)

    def log(self, loss, epoch_num, data_loader, model: SMMModel):

        self.history["loss"].append(LoggingEntity(loss, epoch_num))

        A, B, C, D = model.get_params()
        self.history["A"].append(LoggingEntity(A.cpu().detach().numpy(), epoch_num))
        self.history["B"].append(LoggingEntity(B.cpu().detach().numpy(), epoch_num))
        self.history["C"].append(LoggingEntity(C.cpu().detach().numpy(), epoch_num))
        self.history["D"].append(LoggingEntity(D.cpu().detach().numpy(), epoch_num))

        kernel = model.get_kernel(self.kernel_saving_size).cpu().detach()
        self.history["kernel"].append(LoggingEntity(kernel.detach().numpy(), epoch_num))

        if epoch_num % self.saving_freq == 0:
            self.save()

    def save(self):
        save_dir = os.path.dirname(self.saving_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(self.saving_path, "wb") as f:
            pickle.dump(self.history, f)

    def get_loss_hist(self):
        return [e.entity for e in self.history["loss"]]
