import numpy as np


def swenson(y0, a, increasing=True):
    """
    Returns the nonlinear y parameter from Swenson et al.
    (doi: 10.1063/1.4794808) using the formula from McCarrick et al. when
    applicable (doi: 10.1063/1.4903855).
    Args:
        y0: numpy.ndarray
            Generator detuning in units of linewidths for a = 0.
        a: float
            The nonlinearity parameter.
        increasing: boolean
            The direction of the frequency sweep.
    """
    y = np.empty(y0.shape)
    if increasing:
        # try to compute analytically
        a2 = ((y0 / 3) ** 3 + y0 / 12 + a / 8) ** 2 - ((y0 / 3) ** 2 - 1 / 12) ** 3
        analytic = (a2 >= 0)  # formula breaks down here
        k2 = np.sqrt(a2[analytic])
        a1 = a / 8 + y0 / 12 + k2 + (y0 / 3) ** 3
        k1 = np.sign(a1) * np.abs(a1) ** (1 / 3)  # need the real branch if a1 < 0
        y[analytic] = y0 / 3 + ((y0 / 3) ** 2 - 1 / 12) / k1 + k1
        # use numeric calculation if required
        numeric = np.logical_not(analytic)
        y_numeric = np.empty(numeric.sum())
        for ii, y0_ii in enumerate(y0[numeric]):
            roots = np.roots([4, -4 * y0_ii, 1, -(y0_ii + a)])
            y_numeric[ii] = np.min(roots[np.isreal(roots)].real)
        y0[numeric] = y_numeric
    else:
        # no known analytic formulas for the other direction (yet)
        for ii, y0_ii in enumerate(y0):
            roots = np.roots([4, -4 * y0_ii, 1, -(y0_ii + a)])
            y[ii] = np.max(roots[np.isreal(roots)].real)
    return y