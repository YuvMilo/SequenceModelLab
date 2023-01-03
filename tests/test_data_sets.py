import pytest
import torch


def test_const_signal_shape():
    from src.data_loaders.data_sets.utils.signals import ConstSignalGenerator
    const_sig_gen = ConstSignalGenerator(const=15)
    sig = const_sig_gen.generate(num_signals=100, signal_length=3)
    assert sig.shape == (100, 3)


def test_white_signal_shape():
    from src.data_loaders.data_sets.utils.signals import WhiteSignalGenerator
    const_sig_gen = WhiteSignalGenerator(freq=2)
    sig = const_sig_gen.generate(num_signals=100, signal_length=1000)
    assert sig.shape == (100, 1000)


def test_const_signal_gen():
    from src.data_loaders.data_sets.utils.signals import ConstSignalGenerator
    const_sig_gen = ConstSignalGenerator(const=5)
    sig = const_sig_gen.generate(num_signals=2, signal_length=3)
    assert torch.equal(sig,torch.Tensor([[5, 5, 5], [5, 5, 5]]))


def test_delay_data_set_lag_type_zero():
    from src.data_loaders.data_sets.delay_data_set import DelayedSignalDataset
    from src.data_loaders.data_sets.utils.signals import ConstSignalGenerator
    const_sig_gen = ConstSignalGenerator(const=5)
    ds = DelayedSignalDataset(samples_num=1, seq_length=3,
                              lag_length=1, lag_type="zero",
                              signal_generator=const_sig_gen)
    assert torch.equal(ds.tensors[0], torch.Tensor([[[5], [5], [5]]]))
    assert torch.equal(ds.tensors[1], torch.Tensor([[[0], [5], [5]]]))
