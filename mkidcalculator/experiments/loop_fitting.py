import pickle
import logging
import numpy as np
from collections.abc import Collection

from mkidcalculator.models import S21
from mkidcalculator.io.loop import Loop
from mkidcalculator.io.sweep import Sweep
from mkidcalculator.io.resonator import Resonator
from mkidcalculator.io.utils import _loop_fit_data

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

MAX_REDCHI = 100
FIT_MESSAGE = "loop {} fit: label = '{}', reduced chi squared = {}"


def _get_loops(data):
    if not isinstance(data, Collection):
        data = [data]
    loops = []
    for datum in data:
        if isinstance(datum, Loop):
            loops.append(datum)
        elif isinstance(datum, Resonator):
            loops += datum.loops
        elif isinstance(datum, Sweep):
            for resonator in datum.resonators:
                loops += resonator.loops
        else:
            message = "'data' object ({}) is not a Loop, Resonator, Sweep, or a collection of those objects."
            raise ValueError(message.format(type(data)))
    return loops


def _get_resonators(data):
    if not isinstance(data, Collection):
        data = [data]
    resonators = []
    for datum in data:
        if isinstance(datum, Loop):
            resonators.append(data.resonator)
        elif isinstance(datum, Resonator):
            resonators.append(datum)
        elif isinstance(datum, Sweep):
            resonators += datum.resonators
        else:
            message = "'data' object ({}) is not a Loop, Resonator, Sweep, or a collection of those objects."
            raise ValueError(message.format(type(data)))
    return resonators


def basic_fit(data, label="basic_fit", model=S21, calibration=True, guess_kwargs=None, **lmfit_kwargs):
    """
    Fit the loop using the standard model guess.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        label: string (optional)
            The label to store the fit results under. The default is
            "basic_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        calibration: boolean
            Automatically add 'offset' and 'imbalance' parameters to the guess
            keywords. The default is True, but if a model is used that doesn't
            have those keywords, it should be set to False.
        guess_kwargs: dictionary
            A dictionary of keyword arguments that can overwrite the default
            options for model.guess().
        lmfit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to loop.lmfit()
    """
    # convert file name to loop if needed
    loops = _get_loops(data)
    for loop in loops:
        # make guess
        kwargs = {"imbalance": loop.imbalance_calibration, "offset": loop.offset_calibration} if calibration else {}
        if guess_kwargs is not None:
            kwargs.update(guess_kwargs)
        guess = model.guess(loop.z, loop.f, **kwargs)
        # do fit
        kwargs = {"label": label}
        kwargs.update(lmfit_kwargs)
        loop.lmfit(model, guess, **kwargs)
        log.info(FIT_MESSAGE.format(id(loop), label, loop.lmfit_results[label]['result'].redchi))


def temperature_fit(data, label="temperature_fit", model=S21, **lmfit_kwargs):
    """
    Fit the loop using the two nearest temperature data points of the same
    power in the resonator as guesses. If there are no good guesses, nothing
    will happen.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        label: string (optional)
            The label to store the fit results under. The default is
            "temperature_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        lmfit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to loop.lmfit()
    """
    # convert file name to loop if needed
    loops = _get_loops(data)
    for loop in loops:
        # find good fits from other loop
        good_guesses = []
        temperatures = []
        for potential_loop in loop.resonator.loops:
            if potential_loop is loop:  # don't use fits from this loop
                continue
            if potential_loop.power != loop.power:  # don't use fits with a different power
                continue
            # only use fits that have redchi < MAX_REDCHI
            results_dict = potential_loop.lmfit_results
            if "best" in results_dict.keys() and results_dict['best']['result'].redchi < MAX_REDCHI:
                good_guesses.append(results_dict['best']['result'].params.copy())
                temperatures.append(potential_loop.temperature)
        # fit the two nearest temperature data sets
        indices = np.argsort(np.abs(loop.temperature - np.array(temperatures)))
        for iteration in range(2):
            if iteration < len(indices):
                # pick guess
                guess = good_guesses[indices[iteration]]
                # do fit
                fit_label = label + "_" + str(iteration)
                kwargs = {"label": fit_label}
                kwargs.update(lmfit_kwargs)
                loop.lmfit(model, guess, **lmfit_kwargs)
                log.info(FIT_MESSAGE.format(id(loop), fit_label, loop.lmfit_results[fit_label]['result'].redchi))


def linear_fit(data, label="linear_fit", model=S21, parameter="a_sqrt", **lmfit_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity turned
    off.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        label: string (optional)
            The label to store the fit results under. The default is
            "nonlinear_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        parameter: string (optional)
            The nonlinear parameter name to use.
        lmfit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to loop.lmfit()
    """
    nonlinear_fit(data, label=label, model=model, parameter=(parameter, 0.), vary=False, **lmfit_kwargs)


def nonlinear_fit(data, label="nonlinear_fit", model=S21, parameter=("a_sqrt", 0.05), vary=True, **lmfit_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        label: string (optional)
            The label to store the fit results under. The default is
            "nonlinear_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        parameter: tuple (string, float) (optional)
            The nonlinear parameter name and value to use.
        vary: boolean (optional)
            Determines if the nonlinearity is varied in the fit. The default is
            True.
        lmfit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to loop.lmfit()
    Returns:
        loop: mkidcalculator.Loop
            The loop object that was fit.
    """
    # convert file name to loop if needed
    loops = _get_loops(data)
    for loop in loops:
        # make guess
        if "best" in loop.lmfit_results.keys():
            # only fit if previous fit has been done
            guess = loop.lmfit_results["best"]["result"].params.copy()
            guess[parameter[0]].set(value=parameter[1], vary=vary)
            # do fit
            kwargs = {"label": label}
            kwargs.update(lmfit_kwargs)
            loop.lmfit(model, guess, **lmfit_kwargs)
            log.info(FIT_MESSAGE.format(id(loop), label, loop.lmfit_results[label]['result'].redchi))
        else:
            raise AttributeError("loop does not have a previous fit on which to base the nonlinear fit.")


def multiple_fit(data, model=S21, extra_fits=(temperature_fit, nonlinear_fit, linear_fit), fit_kwargs=None,
                 iterations=2, **basic_fit_kwargs):
    """
    Fit the loops using multiple methods.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        extra_fits: tuple of functions (optional)
            Extra functions to use to try to fit the loops. They must have
            the arguments of basic_fit(). The default is
            (temperature_fit, nonlinear_fit, linear_fit).
        fit_kwargs: dictionary or iterable of dictionaries (optional)
            Extra keyword arguments to send to the extra_fits. The default is
            None and no extra keywords are used. If a single dictionary is
            given, it will be used for all of the extra fits.
        iterations: integer (optional)
            Number of times to run the extra_fits. The default is 2. This is
            useful for when the extra_fits use fit information from other loops
            in the resonator.
        basic_fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the basic_fit function
            before the extra fits are used.
    """
    # parse inputs
    if fit_kwargs is None:
        fit_kwargs = [{}] * len(extra_fits)
    if isinstance(fit_kwargs, dict):
        fit_kwargs = [fit_kwargs] * len(extra_fits)
    # convert file name to resonator if needed
    resonators = _get_resonators(data)
    for resonator in resonators:
        log.info("fitting resonator: {}".format(resonator))
        # fit the resonator
        for iteration in range(iterations):
            log.info("starting iteration: {}".format(iteration))
            # fit loops
            for index, loop in enumerate(resonator.loops):
                log.info("fitting loop: {}".format(index))
                # do the basic fit
                if iteration == 0:
                    basic_fit(loop, model=model, **basic_fit_kwargs)
                # do the extra fits
                for extra_index, fit in enumerate(extra_fits):
                    kwargs = {"label": fit.__name__ + str(iteration), "model": model}
                    kwargs.update(fit_kwargs[extra_index])
                    fit(loop, **kwargs)


def loop_fit_data(data, parameters=("chi2",), label='best', bounds=None, errorbars=None, success=None):
    """
    Collect fit information from Loop fits into arrays.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The fitted loop or loops to extract information from. If Resonator
            or Sweep objects are given all of the contained loops are used.
        parameters: tuple of strings
            The fit parameters to report. "chi2" can be used to retrieve
            the reduced chi squared values. The default is to just return chi2.
        label: string (optional)
            The fit label to use.
        bounds: tuple of numbers or tuples
            The bounds for the parameters. It must be a tuple of the same
            length as the parameters keyword argument. Each element is either
            an upper bound on the parameter or a tuple, e.g. (lower bound,
            upper bound). None can be used as a placeholder to skip a
            parameter. The default is None and no bounds are used.
        errorbars: boolean
            If errorbars is True, only data from loop fits that could compute
            errorbars on the fit parameters is included. If errorbars is False,
            only data from loop fits that could not compute errorbars on the
            fit parameters is included. The default is None, and no filtering
            on the errorbars is done.
        success: boolean
            If success is True, only data from successful loop fits is
            included. If False, only data from failed loop fits is
            included. The default is None, and no filtering on fit success is
            done. Note: fit success is typically a bad indicator on fit
            quality. It only ever fails when something really bad happens.
    Returns:
        outputs: tuple of numpy.ndarray objects
            The outputs in the same order as parameters.
    """
    loops = _get_loops(data)
    return _loop_fit_data(loops, parameters=parameters, label=label, bounds=bounds, errorbars=errorbars,
                          success=success)
