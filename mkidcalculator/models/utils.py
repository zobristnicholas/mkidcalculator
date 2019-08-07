import numpy as np


def scaled_alpha_inv(s_alpha):
    return np.arctan(s_alpha) / np.pi + 1 / 2


def scaled_alpha(alpha):
    return np.tan(np.pi * (alpha - 1 / 2))


def bandpass(data):
    fft_data = np.fft.rfft(data)
    fft_data[:, 0] = 0
    indices = np.array([np.arange(fft_data[0, :].size)] * fft_data[:, 0].size)
    f_data_ind = np.argmax(np.abs(fft_data), axis=-1)[:, np.newaxis]
    fft_data[np.logical_or(indices < f_data_ind - 1, indices > f_data_ind + 1)] = 0
    data_new = np.fft.irfft(fft_data, data[0, :].size)
    return data_new, f_data_ind
