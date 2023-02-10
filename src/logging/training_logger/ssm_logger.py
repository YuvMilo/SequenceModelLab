import os

from src.models.ssm import SMMModel
from src.logging.training_logs.base_log import BaseTrainingLog
from src.logging.training_logger.base_logger import BaseTrainingLogger


class SSMTrainingLogger(BaseTrainingLogger):

    def __init__(self, saving_path: str, saving_freq: int = 500,
                 param_storing_freq: int = 1):
        self.saving_freq = saving_freq
        self.saving_path = saving_path
        self.param_storing_freq = param_storing_freq
        self._training_log = BaseTrainingLog()

    def log(self, loss: float, epoch_num: int,
            model: SMMModel):

        self._training_log.log_training_entity(entity_name="loss", epoch=epoch_num,
                                               value=loss)

        if epoch_num % self.param_storing_freq == 0:
            A, B, C, D = model.get_params()
            self._training_log.log_training_entity(entity_name="A", epoch=epoch_num,
                                                   value=A.cpu().detach().numpy())
            self._training_log.log_training_entity(entity_name="B", epoch=epoch_num,
                                                   value=A.cpu().detach().numpy())
            self._training_log.log_training_entity(entity_name="C", epoch=epoch_num,
                                                   value=A.cpu().detach().numpy())
            self._training_log.log_training_entity(entity_name="D", epoch=epoch_num,
                                                   value=A.cpu().detach().numpy())

        if epoch_num % self.saving_freq == 0:
            self.save()

    def save(self):
        save_dir = os.path.dirname(self.saving_path)
        if save_dir != "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._training_log.save(file_path=self.saving_path)

    def get_loss_hist(self):
        return self._training_log.get_logged_entity_history(entity_name="loss").entities

    # This causes some coupling between "BaseTrainingLog" and this logger,
    # but I think this is essential.
    @property
    def training_log(self) -> BaseTrainingLog:
        return self._training_log
