from typing import Any, List, Callable
import numpy as np
from collections import defaultdict
import pickle
import bz2


class EntityTrainingHistory:

    def __init__(self):
        self._epochs = []
        self._entities = []

    @property
    def epochs(self) -> List[int]:
        return list(np.sort(self._epochs))

    @property
    def entities(self) -> List[Any]:
        return [self._entities[i] for i in np.argsort(self._epochs)]

    def __getitem__(self, key: int) -> Any:
        indexes = np.where(np.array(self._epochs) == key)[0]
        if len(indexes) != 1:
            return None
        else:
            return self._entities[indexes[0]]

    def add_entity(self, epoch, entity):
        self._epochs.append(epoch)
        self._entities.append(entity)


class BaseTrainingLog:

    def __init__(self, running_params: dict = None):
        self.end_results = {}
        self.training_history_entities = defaultdict(EntityTrainingHistory)
        if running_params is None:
            self._running_params = {}
        else:
            self._running_params = running_params.copy()

    def save(self, file_path: str) -> None:
        saving_dict = {
            "end_results": self.end_results,
            "training_history_entities": self.training_history_entities,
            "running_params": self._running_params
        }

        with bz2.BZ2File(file_path, "wb") as f:
            pickle.dump(saving_dict, f)

    def load(self, file_path: str) -> None:
        with bz2.BZ2File(file_path, "rb") as f:
            saving_dict = pickle.load(f)
        self.end_results = saving_dict["end_results"]
        self.training_history_entities = saving_dict["training_history_entities"]
        self._running_params = saving_dict["running_params"]

    def log_training_entity(self, entity_name: str, epoch: int, value: Any) -> None:
        self.training_history_entities[entity_name].add_entity(epoch, value)

    def log_end_result(self, end_result_name: str, value: Any) -> None:
        self.end_results[end_result_name] = value

    def add_entity_history_by_augmentation(self, entity_name: str, parameter: List[str],
                                           augmentation_func: Callable[..., Any]):
        epochs_with_params = set(self.training_history_entities[parameter[0]].epochs)
        for param in parameter[1:]:
            cur_epochs = set(self.training_history_entities[param].epochs)
            epochs_with_params = epochs_with_params.intersection(cur_epochs)

        for epoch in epochs_with_params:
            cur_params = [self.training_history_entities[param][epoch]
                          for param in parameter]
            self.training_history_entities[entity_name].add_entity(
                epoch=epoch,
                entity=augmentation_func(*cur_params)
            )

    def get_logged_entity_history(self, entity_name: str) -> EntityTrainingHistory:
        return self.training_history_entities[entity_name]

    def get_end_result(self, end_result_name: str) -> Any:
        return self.end_results[end_result_name]

    @property
    def logged_training_entities(self) -> List[str]:
        return [key for key in self.training_history_entities]

    @property
    def logged_end_results(self) -> List[str]:
        return [key for key in self.end_results]

    @property
    def running_params(self) -> dict:
        return self._running_params

    def __bool__(self) -> bool:
        if len(self.end_results) != 0:
            return True
        if len(self.training_history_entities) != 0:
            return True
        return False
