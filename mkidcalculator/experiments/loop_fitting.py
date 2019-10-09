import pickle
import logging
import numpy as np

from mkidcalculator.models import S21
from mkidcalculator.io.loop import Loop
from mkidcalculator.io.resonator import Resonator

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

MAX_REDCHI = 100


def _load_loop(loop, **load_kwargs):
    if isinstance(loop, str):
        if load_kwargs:
            loop = Loop.from_file(loop, **load_kwargs)
        else:
            try:
                loop = Loop.from_pickle(loop)
            except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError, AssertionError):
                loop = Loop.from_file(loop)
    return loop


def _load_resonator(resonator, **load_kwargs):
    # convert file name to resonator if needed
    if isinstance(resonator, str):
        if load_kwargs:
            resonator = Resonator.from_file(resonator, **load_kwargs)
        else:
            try:
                resonator = Resonator.from_pickle(resonator)
            except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError, AssertionError):
                resonator = Resonator.from_file(resonator)
    return resonator


def basic_fit(loop, label="basic_fit", model=S21, **load_kwargs):
    """
    Fit the loop using the standard model guess.
    Args:
        loop: string or mkidcalculator.Loop
            The loop object to fit. If a string, the loop is loaded from either
            Loop.from_pickle() or Loop.from_file().
        label: string (optional)
            The label to store the fit results under. The default is
            "basic_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        load_kwargs: optional keyword arguments
            Keyword arguments to send to Loop.from_file(). Loop.from_pickle()
            will not be attempted if kwargs are given.
    Returns:
        loop: mkidcalculator.Loop
            The loop object that was fit.
    """
    # convert file name to loop if needed
    loop = _load_loop(loop, **load_kwargs)
    # make guess
    guess = model.guess(loop.z, loop.f, loop.imbalance_calibration, loop.offset_calibration)
    # do fit
    loop.lmfit(model, guess, label=label)
    return loop


def temperature_fit(loop, label="temperature_fit", model=S21, **load_kwargs):
    """
    Fit the loop using the two nearest temperature data points of the same
    power in the resonator as guesses. If there are no good guesses, nothing
    will happen.
    Args:
        loop: string or mkidcalculator.Loop
            The loop object to fit. If a string, the loop is loaded from either
            Loop.from_pickle() or Loop.from_file().
        label: string (optional)
            The label to store the fit results under. The default is
            "temperature_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        load_kwargs: optional keyword arguments
            Keyword arguments to send to Loop.from_file(). Loop.from_pickle()
            will not be attempted if kwargs are given.
    Returns:
        loop: mkidcalculator.Loop
            The loop object that was fit.
    """
    # convert file name to loop if needed
    loop = _load_loop(loop, **load_kwargs)
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
            loop.lmfit(model, guess, label=label + "_" + str(iteration))
    return loop


def linear_fit(loop, label="linear_fit", model=S21, parameter="a_sqrt", **load_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity turned
    off.
    Args:
        loop: string or mkidcalculator.Loop
            The loop object to fit. If a string, the loop is loaded from either
            Loop.from_pickle() or Loop.from_file().
        label: string (optional)
            The label to store the fit results under. The default is
            "nonlinear_fit".
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        parameter: string (optional)
            The nonlinear parameter name to use.
        load_kwargs: optional keyword arguments
            Keyword arguments to send to Loop.from_file(). Loop.from_pickle()
            will not be attempted if kwargs are given.
    Returns:
        loop: mkidcalculator.Loop
            The loop object that was fit.
    """
    loop = nonlinear_fit(loop, label=label, model=model, parameter=(parameter, 0.), vary=False, **load_kwargs)
    return loop


def nonlinear_fit(loop, label="nonlinear_fit", model=S21, parameter=("a_sqrt", 0.05), vary=True, **load_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity.
    Args:
        loop: string or mkidcalculator.Loop
            The loop object to fit. If a string, the loop is loaded from either
            Loop.from_pickle() or Loop.from_file().
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
        load_kwargs: optional keyword arguments
            Keyword arguments to send to Loop.from_file(). Loop.from_pickle()
            will not be attempted if kwargs are given.
    Returns:
        loop: mkidcalculator.Loop
            The loop object that was fit.
    """
    # convert file name to loop if needed
    loop = _load_loop(loop, **load_kwargs)
    # make guess
    if "best" in loop.lmfit_results.keys():
        # only fit if previous fit has been done
        guess = loop.lmfit_results["best"]["result"].params.copy()
        guess[parameter[0]].set(value=parameter[1], vary=vary)
        # do fit
        loop.lmfit(model, guess, label=label)
    else:
        raise AttributeError("loop does not have a previous fit on which to base the nonlinear fit.")
    return loop


def resonator_fit(resonator, model=S21, extra_fits=(temperature_fit, nonlinear_fit, linear_fit), fit_kwargs=(),
                  iterations=2, **load_kwargs):
    """
    Fit all of the loops in a resonator.
    Args:
        resonator: string or mkidcalculator.Resonator
            The resonator object to use for the fit. If a string, the resonator
            is loaded from either Resonator.from_pickle() or
            Resonator.from_file().
        model: class (optional)
            A model class to use for the fit. The default is
            mkidcalculator.models.S21.
        extra_fits: tuple of functions (optional)
            Extra functions to use to try to fit the loops. They must have
            the arguments of basic_fit(). The default is
            (temperature_fit, nonlinear_fit, linear_fit).
        fit_kwargs: tuple of dictionaries (optional)
            Extra keyword arguments to send to the extra_fits. The default is
            an empty tuple which corresponds to no extra keyword arguments.
        iterations: integer (optional)
            Number of times to run the extra_fits. The default is 2. This is
            useful for when the extra_fits use fit information from other loops
            in the resonator.
        load_kwargs: optional keyword arguments
            Keyword arguments to send to Resonator.from_file().
            Resonator.from_pickle() will not be attempted if kwargs are given.
    Returns:
        resonator: mkidcalculator.Resonator
            The resonator object that was fit.
    """
    # parse inputs
    if not fit_kwargs:
        fit_kwargs = [{}] * len(extra_fits)
    # convert file name to resonator if needed
    resonator = _load_resonator(resonator, **load_kwargs)
    log.info("fitting resonator: {}".format(resonator))
    # fit the resonator
    for iteration in range(iterations):
        log.info("starting iteration: {}".format(iteration))
        # fit loops
        for index, loop in enumerate(resonator.loops):
            log.info("fitting loop: {}".format(index))
            # do the basic fit
            if iteration == 0:
                basic_fit(loop, model=model)
            # do the extra fits
            for extra_index, fit in enumerate(extra_fits):
                kwargs = {"label": fit.__name__ + str(iteration), "model": model}
                kwargs.update(fit_kwargs[extra_index])
                fit(loop, **kwargs)
    # log bad fits
    for index, loop in enumerate(resonator.loops):
        redchi = loop.lmfit_results['best']['result'].redchi
        if redchi > MAX_REDCHI:
            log.warning("loop {} failed to fit with redchi of {}".format(index, redchi))
    return resonator
