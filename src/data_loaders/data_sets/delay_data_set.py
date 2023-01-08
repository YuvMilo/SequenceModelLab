import torch

from src.data_loaders.data_sets.utils.signals import ConstSignalGenerator


class DelayedSignalDataset(torch.utils.data.TensorDataset):

    def __init__(self, samples_num=1, seq_length=10000, lag_length=1000,
                 signal_generator=None, lag_type="zero"):
        assert lag_length < seq_length

        if signal_generator is None:
            signal_generator = ConstSignalGenerator(0)

        X = signal_generator.generate(num_signals=samples_num,
                                      signal_length=seq_length)
        X = X.unsqueeze(-1)

        if lag_type == "zero":
            Y = torch.zeros(X.shape)
            Y[:, lag_length:, :] = X[:, :-lag_length, :]
        else:
            raise NotImplementedError("lag_type {} not implemented".format(lag_type))

        super().__init__(X, Y)


class DelayedSignalDatasetRegenerated(torch.utils.data.TensorDataset):

    def __init__(self, samples_num=1, seq_length=10000, lag_length=1000,
                 signal_generator=None, lag_type="zero"):
        assert lag_length < seq_length
        assert lag_type in ["zero"]

        if signal_generator is None:
            signal_generator = ConstSignalGenerator(0)

        self.signal_generator = signal_generator
        self.samples_num = samples_num
        self.seq_length = seq_length
        self.lag_length = lag_length
        self.lag_type = lag_type
        super().__init__()

    def __getitem__(self, index):
        X = self.signal_generator.generate(num_signals=1,
                                           signal_length=self.seq_length)
        X = X.unsqueeze(-1)

        if self.lag_type == "zero":
            Y = torch.zeros(X.shape)
            Y[:, self.lag_length:, :] = X[:, :-self.lag_length, :]

        return X[0, :, :], Y[0, :, :]

    def __len__(self):
        return self.samples_num
