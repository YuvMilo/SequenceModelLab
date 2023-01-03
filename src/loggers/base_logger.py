from collections import defaultdict


class BaseLogger:

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, loss, epoch, model=None):
        self.history["loss"].append(loss)
