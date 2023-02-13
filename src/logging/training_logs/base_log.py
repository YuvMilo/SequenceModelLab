from typing import Any, List, Callable
import numpy as np
from collections import defaultdict
import pickle
import lz4.frame
from src.utils.imutable_dict import ImmutableDict


class EntityTrainingHistory:

    def __init__(self):
        self._epochs = []
        self._entities = []
        self.sorted = True

    @property
    def epochs(self) -> List[int]:
        if not self.sorted:
            self.sort()
        return self._epochs.copy()

    @property
    def entities(self) -> List[Any]:
        if not self.sorted:
            self.sort()
        return self._entities.copy()

    def __getitem__(self, key: int) -> Any:
        if not self.sorted:
            self.sort()

        i1 = np.searchsorted(self._epochs, key, 'left')
        i2 = np.searchsorted(self._epochs, key, 'right')
        if np.any((i2 - i1) != 1):
            return None
        else:
            return np.array(self._entities)[i1]

    def add_entity(self, epoch, entity):
        self._epochs.append(epoch)
        self._entities.append(entity)
        self.sorted = False

    def sort(self):
        self._entities = [self._entities[i] for i in np.argsort(self._epochs)]
        self._epochs = list(np.sort(self._epochs))
        self.sorted = True


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

        with lz4.frame.open(file_path, "wb") as f:
            pickle.dump(saving_dict, f)

    def load(self, file_path: str) -> None:
        with lz4.frame.open(file_path, "rb") as f:
            saving_dict = pickle.load(f)
        self.end_results = saving_dict["end_results"]
        self.training_history_entities = saving_dict["training_history_entities"]
        self._running_params = saving_dict["running_params"]

    def log_training_entity(self, entity_name: str, epoch: int, value: Any) -> None:
        self.training_history_entities[entity_name].add_entity(epoch, value)

    def log_end_result(self, end_result_name: str, value: Any) -> None:
        self.end_results[end_result_name] = value

    def add_entity_history_by_augmentation(self, entity_name: str,
                                           parameters: List[str],
                                           augmentation_func: Callable[..., Any]):
        epochs_with_params = set(self.training_history_entities[parameters[0]].epochs)
        for param in parameters[1:]:
            cur_epochs = set(self.training_history_entities[param].epochs)
            epochs_with_params = epochs_with_params.intersection(cur_epochs)
        epochs_with_params = sorted(list(epochs_with_params))

        # This is quicker then searching for training_history_entities[param][epoch]
        # foreach epoch
        params_values = []
        for param in parameters:
            cur_param_value = self.training_history_entities[param][epochs_with_params]
            params_values.append(cur_param_value)

        for i, epoch in enumerate(epochs_with_params):
            cur_params = [params_values[param_idx][i]
                          for param_idx in range(len(parameters))]
            self.training_history_entities[entity_name].add_entity(
                epoch=epoch,
                entity=augmentation_func(*cur_params)
            )

    def add_entity_end_result_by_augmentation(self, entity_name: str,
                                              parameters: List[str],
                                              augmentation_func: Callable[..., Any]):

        epochs_with_params = set(self.training_history_entities[parameters[0]].epochs)
        for param in parameters[1:]:
            cur_epochs = set(self.training_history_entities[param].epochs)
            epochs_with_params = epochs_with_params.intersection(cur_epochs)
        epochs_with_params = sorted(list(epochs_with_params))

        params_values = []
        for param in parameters:
            cur_param_value = self.training_history_entities[param][epochs_with_params]
            params_values.append(cur_param_value)

        self.end_results[entity_name] = augmentation_func(
            *([epochs_with_params] + params_values)
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
    def running_params(self) -> ImmutableDict:
        return ImmutableDict(self._running_params)

    def __bool__(self) -> bool:
        for param in self.logged_training_entities:
            if len(self.get_logged_entity_history(param).entities) > 1:
                return True
        return False
