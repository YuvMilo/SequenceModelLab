import os
from typing import List

import torch

from src.models.ssm import SMMModel
from src.logging.training_logs.base_log import BaseTrainingLog
from src.logging.training_logger.base_logger import BaseTrainingLogger


class SSMTrainingLogger(BaseTrainingLogger):

    def __init__(self, saving_path: str = None, saving_freq: int = 1e6,
                 param_storing_freq: int = 1,
                 running_params: dict = None):
        self.saving_freq = saving_freq
        self.saving_path = saving_path
        self.param_storing_freq = param_storing_freq
        self._training_log = BaseTrainingLog(running_params=running_params)
        self.got_logged = False

    def log(self, loss: float, epoch_num: int,
            model: SMMModel) -> None:
        self.got_logged = True

        self._training_log.log_training_entity(entity_name="loss", epoch=epoch_num,
                                               value=loss)

        if epoch_num % self.param_storing_freq == 0:
            A, B, C, D = model.get_params()
            if type(A) is torch.Tensor:
                self._training_log.log_training_entity(
                    entity_name="A", epoch=epoch_num,
                    value=A.cpu().detach().numpy()
                )
            elif type(A) is list:
                self._training_log.log_training_entity(
                    entity_name="A", epoch=epoch_num,
                    value=[a.cpu().detach().numpy() for a in A]
                )
            else:
                raise
            if type(B) is torch.Tensor:
                self._training_log.log_training_entity(
                    entity_name="B", epoch=epoch_num,
                    value=B.cpu().detach().numpy()
                )
            elif type(B) is list:
                self._training_log.log_training_entity(
                    entity_name="B", epoch=epoch_num,
                    value=[b.cpu().detach().numpy() for b in B]
                )
            else:
                raise
            if type(C) is torch.Tensor:
                self._training_log.log_training_entity(
                    entity_name="C", epoch=epoch_num,
                    value=C.cpu().detach().numpy().copy()
                )
            elif type(C) is list:
                self._training_log.log_training_entity(
                    entity_name="A", epoch=epoch_num,
                    value=[c.cpu().detach().numpy() for c in C]
                )
            else:
                raise
            if type(D) is torch.Tensor:
                self._training_log.log_training_entity(
                    entity_name="D", epoch=epoch_num,
                    value=D.cpu().detach().numpy()
                )
            elif type(D) is list:
                self._training_log.log_training_entity(
                    entity_name="A", epoch=epoch_num,
                    value=[d.cpu().detach().numpy() for d in D]
                )
            else:
                raise

        if (epoch_num + 1) % self.saving_freq == 0:
            self.save()

    def save(self) -> None:
        if not self.got_logged:
            return

        if self.saving_path is None:
            return

        save_dir = os.path.dirname(self.saving_path)
        if save_dir != "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._training_log.save(file_path=self.saving_path)

    def get_loss_hist(self) -> List[float]:
        return self._training_log.get_logged_entity_history(entity_name="loss").entities

    # This causes some coupling between "BaseTrainingLog" and this logger,
    # but I think this is essential.
    @property
    def training_log(self) -> BaseTrainingLog:
        return self._training_log

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def __del__(self):
        self.save()
