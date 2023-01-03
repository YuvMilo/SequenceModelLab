import torch
from torch.utils.data import DataLoader

from src.data_loaders.data_sets.delay_data_set import DelayedSignalDataset, DelayedSignalDatasetRegenerated
from src.loggers.base_logger import BaseLogger
from src.models.linear_sequence_models.LRNN import OneLayerDiagTransitionLinearRNN
from src.models.linear_sequence_models.linear_S4 import LS4D,LS4DSimplified
from src.models.linear_sequence_models.LSSM import LSSMWithS4Parameterization
from src.data_loaders.data_sets.utils.signals import ConstSignalGenerator
from src.data_loaders.data_sets.utils.signals import WhiteSignalGenerator
from src.training import train
from src.loss_function.delay_loss_function import delay_l2


def main():
    lag = 20
    hidden_size = 500
    seq_len = 100
    criterion = delay_l2(lag)
    signal_generator = WhiteSignalGenerator(freq=1, dt=0.05)
    #ds = DelayedSignalDatasetRegenerated(lag_length=lag, samples_num=10,
    #                          seq_length=seq_len, signal_generator=signal_generator)
    ds = DelayedSignalDataset(lag_length=lag, samples_num=10,
                              seq_length=seq_len, signal_generator=signal_generator)
    dl = DataLoader(ds, batch_size=1)

    # model = OneLayerDiagTransitionLinearRNN(1, hidden_size, 1)
    # model = LS4DSimplified(d_model=1, N=hidden_size, L=seq_len)
    model = LSSMWithS4Parameterization(d_model=1, N=hidden_size, L=seq_len)
    logger = BaseLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    train(model=model,
          dl=dl,
          logger=logger,
          criterion=criterion,
          num_epochs=1000,
          optimizer=optimizer)

    pass


if __name__ == "__main__":
    main()
