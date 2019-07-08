import numpy as np


def scaled_alpha_inv(s_alpha):
    return np.arctan(s_alpha) / np.pi + 1 / 2


def scaled_alpha(alpha):
    return np.tan(np.pi * (alpha - 1 / 2))
