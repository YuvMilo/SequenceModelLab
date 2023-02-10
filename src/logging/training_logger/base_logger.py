from collections import defaultdict


class BaseTrainingLogger:

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, loss, epoch_num, model):
        self.history["loss"].append(loss)

    def save(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()