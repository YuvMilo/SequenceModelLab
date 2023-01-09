import numpy as np
import torch


class SignalGenerator:
    def __init__(self):
        raise NotImplementedError("Subclass should implement this")

    def generate(self, num_signal, signal_length, signal_dim=1):
        raise NotImplementedError("Subclass should implement this")


class ConstSignalGenerator(SignalGenerator):
    def __init__(self, const=0):
        self.const = const

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.ones([num_signals, signal_length])
        return torch.fill(signals, self.const)


class NormalNoiseCumSignalGenerator(SignalGenerator):
    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.randn([num_signals, signal_length])
        signals = torch.cumsum(signals, dim=1)
        return signals


class NormalNoiseSignalGenerator(SignalGenerator):
    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.randn([num_signals, signal_length])
        return signals


class WhiteSignalGenerator(SignalGenerator):
    def __init__(self, dt=1e-2, freq=1.0):
        self.dt = dt
        self.freq = freq

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("WhiteSignalGenerator only supports signal dim 1")

        signals = [self._white_signal(period=signal_length * self.dt,
                                      dt=self.dt,
                                      freq=self.freq)
                   for i in range(num_signals)]
        signals = np.array(signals)
        signals = torch.Tensor(signals)
        return signals

    def _white_signal(self, period, dt, freq, rms=0.5, batch_shape=()):
        if freq is not None and freq < 1. / period:
            raise ValueError(f"Make ``{freq=} >= 1. / {period=}`` to produce a"
                             f" non-zero signal", )

        nyquist_cutoff = 0.5 / dt
        if freq > nyquist_cutoff:
            raise ValueError("freq must not exceed the Nyquist frequency")

        n_coefficients = int(np.ceil(period / dt / 2.))
        shape = batch_shape + (n_coefficients + 1,)
        sigma = rms * np.sqrt(0.5)
        coefficients = 1j * np.random.normal(0., sigma, size=shape)
        coefficients[..., -1] = 0.
        coefficients += np.random.normal(0., sigma, size=shape)
        coefficients[..., 0] = 0.

        set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
        coefficients *= (1 - set_to_zero)
        power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float)
                                   / n_coefficients)
        if power_correction > 0.:
            coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)
        signal = np.fft.irfft(coefficients, axis=-1)
        signal = signal - signal[..., :1]  # Start from 0
        return signal
