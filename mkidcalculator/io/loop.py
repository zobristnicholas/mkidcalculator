import os
import copy
import logging
import lmfit as lm
import numpy as np
from operator import itemgetter
import scipy.stats as stats
from scipy.interpolate import make_interp_spline


from mkidcalculator.io.noise import Noise
from mkidcalculator.io.pulse import Pulse
from mkidcalculator.io.data import AnalogReadoutLoop, AnalogReadoutNoise, AnalogReadoutPulse
from mkidcalculator.io.utils import (ev_nm_convert, lmfit, sort_and_fix, setup_axes, finalize_axes, get_plot_model,
                                     dump, load)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Loop:
    """A class for manipulating resonance loop scattering parameter data."""
    def __init__(self):
        # loop data
        self._data = AnalogReadoutLoop()  # dummy class replaced by from_file()
        self._mask = None
        # resonator reference
        self._resonator = None
        # noise and pulse classes
        self.noise = []
        self.f_bias_noise = []  # for bias frequency of each noise data set
        self.pulses = []
        self.f_bias_pulses = []  # for bias frequency of each pulse data set
        self.max_energy_pulses = []  # for known line energies for each pulse data set
        # internal variables
        self._power_calibration = 0
        # analysis results
        self.lmfit_results = {}
        self.emcee_results = {}
        # directory of the saved data
        self._directory = None
        # response calibrations
        self.energy_calibration = None
        self._energy_calibration_type = None
        self._response_avg = None
        self._response_energies = None
        # energy calibrations
        self.phase_calibration = None
        self.amplitude_calibration = None
        self._phase_avg = None
        self._phase_energies = None
        self._amplitude_avg = None
        self._amplitude_energies = None
        # template calibration
        self.template_fft = None
        self._template_size = None
        log.debug("Loop object created. ID: {}".format(id(self)))

    @property
    def z(self):
        """The complex scattering parameter for the resonance loop."""
        return self._data['z']

    @property
    def f(self):
        """The frequencies corresponding to the complex scattering parameter."""
        return self._data['f']

    @property  # @property so that self.f not put into memory on load
    def f_center(self):
        """
        The median frequency in loop.f. This is a useful rough proxy for the
        resonant frequency that depends only on the data and not the fit.
        """
        return np.median(self.f)

    @property
    def imbalance_calibration(self):
        """A MxN complex array containing beating IQ mixer data on the rows."""
        return self._data['imbalance']

    @property
    def offset_calibration(self):
        """The mixer offsets corresponding to the complex scattering parameter."""
        return self._data['offset']

    @property
    def power_calibration(self):
        """
        The calibrated power in dBm at the resonator for zero attenuation. This
        value is often unknown so it is initialized to zero. If it is known, it
        can be set directly (e.g. self.power_calibration = value). This value
        is used primarily in the self.power attribute.
        """
        return self._power_calibration

    @power_calibration.setter
    def power_calibration(self, calibration):
        self._power_calibration = calibration

    @property
    def metadata(self):
        """A dictionary containing metadata about the loop."""
        return self._data['metadata']

    @property
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def power(self):
        """
        The power in dBm at the resonator. This value is only relative if
        self.power_calibration has not been set.
        """
        return self.power_calibration - self.attenuation

    @property
    def field(self):
        """The field value at the resonator."""
        return self._data['field']

    @property
    def temperature(self):
        """The temperature at the resonator."""
        return self._data['temperature']

    @property
    def resolving_powers(self):
        """Returns a list of resolving powers for the pulses."""
        return [pulse.resolving_power for pulse in self.pulses]

    @property
    def resonator(self):
        """
        A settable property that contains the Resonator object that this loop
        has been assigned to. If the resonator has not been set, it will raise
        an AttributeError.
        """
        if self._resonator is None:
            raise AttributeError("The resonator object for this loop has not been set yet.")
        return self._resonator

    @resonator.setter
    def resonator(self, resonator):
        self._resonator = resonator

    @property
    def mask(self):
        """
        A settable property that contains a boolean array that can select
        frequency indices from loop.z and loop.f.
        """
        if self._mask is None:
            self._mask = np.ones(self.f.size, dtype=bool)
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def mask_from_center(self, df, center=None):
        """
        Mask the loop data using a frequency window and a center frequency.
        Args:
            df: float
                The frequency window in units of loop.f.
            center: float, string (optional)
                The center frequency to select around. If not provided, the
                median of loop.f will be used as the center. If either "min" or
                "max" are given, the min or max transmission point will be used
                as the center.
        """
        if center is not None:
            if isinstance(center, str):
                if center.lower().startswith("min"):
                    f0 = self.f[np.argmin(np.abs(self.z))]
                elif center.lower().startswith("max"):
                    f0 = self.f[np.argmax(np.abs(self.z))]
                else:
                    raise ValueError("'center' must be in [min, max, None] or be a float")
            else:
                f0 = center
        else:
            f0 = np.median(self.f)
        self.mask = self.mask & (self.f >= f0 - df / 2) & (self.f <= f0 + df / 2)

    def mask_from_bounds(self, lower=None, upper=None):
        """
        Mask the loop data using an upper or lower bound on the frequency.
        Args:
            lower: float (optional)
                The lowest frequency to include in the data.
            upper: float (optional)
                The highest frequency to include in the data.
        """
        if lower is not None:
            self.mask = self.mask & (self.f >= lower)
        if upper is not None:
            self.mask = self.mask & (self.f <= upper)

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        dump(self, file_name)
        log.info("saved loop as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Loop class from the pickle file 'file_name'."""
        loop = load(file_name)
        assert isinstance(loop, cls), "'{}' does not contain a Loop class.".format(file_name)
        log.info("loaded loop from '{}'".format(file_name))
        return loop

    @classmethod
    def from_python(cls, z, f, attenuation, field, temperature, imbalance_calibration=None, offset=None, metadata=None):
        """
        Returns a Loop class from python data.
        Args:
            z: numpy.ndarray
                The complex scattering parameter for the resonance loop.
            f: numpy.ndarray
                The frequencies corresponding to the complex scattering
                parameter.
            attenuation: float
                The DAC attenuation used for the data set.
            field: float
                The field value at the resonator.
            temperature: float
                The temperature at the resonator.
            imbalance_calibration: numpy.ndarray (optional)
                A MxN complex array containing beating IQ mixer data on the
                rows.
            offset: numpy.ndarray (optional)
                The mixer offsets corresponding to the complex scattering
                parameter.
            metadata: dictionary (optional)
                A dictionary containing metadata about the loop.
        Returns:
            loop: mkidcalculator.Loop object
                A loop class with the python data loaded.
        """
        loop = cls()
        loop._data = {"z": z, "f": f, "imbalance": imbalance_calibration, "offset": offset, "metadata": metadata,
                      "attenuation": attenuation, "field": field, "temperature": temperature}
        return loop

    def add_pulses(self, pulses, sort=True):
        """
        Add Pulse objects to the loop.
        Args:
            pulses: Pulse class or iterable of Pulse classes
                The pulse objects that are to be added to the Loop.
            sort: boolean (optional)
                Sort the pulse list by its bias frequency and then its
                maximum energy. The default is True. If False, the order of the
                pulse list is preserved.
        """
        if isinstance(pulses, Pulse):
            pulses = [pulses]
        # append pulse data
        for p in pulses:
            p.loop = self  # set the loop here instead of from_file because add_pulses() can be used independently
            self.pulses.append(p)
            self.f_bias_pulses.append(p.f_bias)
            self.max_energy_pulses.append(np.max(p.energies))
        # sort
        if sort and self.pulses:
            lp = zip(*sorted(zip(self.f_bias_pulses, self.max_energy_pulses, self.pulses), key=itemgetter(0, 1)))
            self.f_bias_pulses, self.max_energy_pulses, self.pulses = (list(t) for t in lp)

    def remove_pulses(self, indices):
        """
        Remove pulses from the loop.
        Args:
            indices: integer or iterable of integers
                The indices in resonator.pulses that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.pulses.pop(ii)
            self.f_bias_pulses.pop(ii)
            self.max_energy_pulses.pop(ii)

    def add_noise(self, noise, sort=True):
        """
        Add Noise objects to the loop.
        Args:
            noise: Noise class or iterable of Noise classes
                The noise objects that are to be added to the Loop.
            sort: boolean (optional)
                Sort the noise list by its bias frequency. The default is
                True. If False, the order of the noise list is preserved.
        """
        if isinstance(noise, Noise):
            noise = [noise]
        # append noise data
        for n in noise:
            n.loop = self  # set the noise here instead of from_file because add_noise() can be used independently
            self.noise.append(n)
            self.f_bias_noise.append(n.f_bias)
        # sort
        if sort and self.noise:
            self.f_bias_noise, self.noise = (list(t) for t in
                                             zip(*sorted(zip(self.f_bias_noise, self.noise), key=itemgetter(0))))

    def remove_noise(self, indices):
        """
        Remove noise from the loop.
        Args:
            indices: integer or iterable of integers
                The indices in resonator.noise that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.noise.pop(ii)
            self.f_bias_noise.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Pulse and Noise objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for pulse in self.pulses:
            pulse.free_memory(directory=directory)
        for noise in self.noise:
            noise.free_memory(directory=directory)
        try:
            self._data.free_memory()
        except AttributeError:
            pass

    @classmethod
    def from_file(cls, loop_file_name, noise_file_names=(), pulse_file_names=(), data=AnalogReadoutLoop, sort=True,
                  noise_data=None, pulse_data=None, channel=None, noise_kwargs=None, pulse_kwargs=None, **kwargs):
        """
        Loop class factory method that returns a Loop() with the loop, noise
        and pulse data loaded.
        Args:
            loop_file_name: string
                The file name for the loop data.
            noise_file_names: tuple (optional)
                Tuple of file name strings for the noise data. The default is
                to not load any noise data.
            pulse_file_names: tuple (optional)
                Tuple of file name strings for the pulse data. The default is
                to not load any pulse data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Loop class. The
                default is the AnalogReadoutLoop class, which interfaces
                with the data products from the analogreadout module.
            sort: boolean (optional)
                Sort the noise data and pulse data lists by their bias
                frequencies. The default is True. If False, the order of the
                noise and pulse file names is preserved.
            noise_data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the None. If not None, this value overloads any
                'data' keyword in noise_kwargs.
            pulse_data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the None. If not None, this value overloads any
                'data' keyword in pulse_kwargs.
            channel: integer (optional)
                Optional channel index which gets added to all of the kwarg
                dictionaries under the key 'channel'. When the default, None,
                is passed, nothing is added to the dictionaries.
            noise_kwargs: tuple (optional)
                Tuple of dictionaries for extra keyword arguments to send to
                noise_data. The order and length correspond to
                noise_file_names. The default is None, which is equivalent to
                a tuple of {'data': AnalogReadoutNoise}. The data keyword is
                always set to this value unless specified.
            pulse_kwargs: tuple (optional)
                Tuple of dictionaries for extra keyword arguments to send to
                pulse_data. The order and length correspond to
                pulse_file_names. The default is None, which is equivalent to
                a tuple of {'data': AnalogReadoutPulse}. The data keyword is
                always set to this value unless specified.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to loop_data.
        Returns:
            loop: object
                A Loop() object containing the loaded data.
        """
        # create loop
        loop = cls()
        # update dictionaries
        if noise_kwargs is None:
            noise_kwargs = [{} for _ in range(len(noise_file_names))]
        if pulse_kwargs is None:
            pulse_kwargs = [{} for _ in range(len(pulse_file_names))]
        if noise_data is not None:
            for kws in noise_kwargs:
                kws.update({'data': noise_data})
        if pulse_data is not None:
            for kws in pulse_kwargs:
                kws.update({'data': pulse_data})
        if channel is not None:
            kwargs.update({"channel": channel})
            for kws in noise_kwargs:
                kws.update({"channel": channel})
            for kws in pulse_kwargs:
                kws.update({"channel": channel})
        # load loop
        loop._data = data(loop_file_name, **kwargs)
        # load noise
        noise = []
        for index, noise_file_name in enumerate(noise_file_names):
            noise.append(Noise.from_file(noise_file_name, **noise_kwargs[index]))
        loop.add_noise(noise, sort=sort)
        # load pulses
        pulses = []
        for index, pulse_file_name in enumerate(pulse_file_names):
            pulses.append(Pulse.from_file(pulse_file_name, **pulse_kwargs[index]))
        loop.add_pulses(pulses, sort=sort)
        return loop

    def lmfit(self, model, guess, label='default', use_mask=True, keep=True, residual_args=(), residual_kwargs=None,
              **kwargs):
        """
        Compute a least squares fit using the supplied residual function and
        guess. The result and other useful information is stored in
        self.lmfit_results[label].
        Args:
            model: object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first three arguments must
                be a lmfit.Parameters object, the complex scattering parameter,
                and the corresponding frequencies. Other arguments can be
                passed in through the residual_args and residual_kwargs
                arguments.
            guess: lmfit.Parameters object
                A parameters object containing starting values (and bounds if
                desired) for all of the parameters needed for the residual
                function.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.lmfit_results dictionary. The default is 'default'.
            use_mask: boolean (optional)
                Use the mask to select the frequency and complex transmission
                data. The default is True.
            keep: boolean (optional)
                Store the fit result in the object. The default is True. If
                False, the fit will only be stored if it is the best so far.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first three. The default is an empty tuple.
            residual_kwargs: dictionary (optional)
                A dictionary of arguments to be passed to the residual
                function. The default is None, which corresponds to an empty
                dictionary.
            kwargs: optional keyword arguments
                Additional keyword arguments are sent to the
                lmfit.Minimizer.minimize() method.
        Returns:
            result: lmfit.MinimizerResult
                An object containing the results of the minimization. It is
                also stored in self.lmfit_results[label]['result'].
        """
        if use_mask:
            residual_args = (self.z[self.mask], self.f[self.mask], *residual_args)
        else:
            residual_args = (self.z, self.f, *residual_args)
        result = lmfit(self.lmfit_results, model, guess, label=label, keep=keep, residual_args=residual_args,
                       residual_kwargs=residual_kwargs, **kwargs)
        return result

    def emcee(self, model, label='default', residual_args=(), residual_kwargs=None, **kwargs):
        """
        Compute a MCMC using the supplied log likelihood function. The result
        and other useful information is stored in self.emcee_results[label].
        Args:
            model: module or object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first three arguments must
                be a lmfit.Parameters object, the complex scattering parameter,
                and the corresponding frequencies. Other arguments can be
                passed in through the residual_args and residual_kwargs
                arguments.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.emcee_results dictionary. A corresponding fit must already
                exist in the self.lmfit_results dictionary. The default is
                'default'.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first three. The default is an empty tuple.
            residual_kwargs: dictionary (optional)
                A dictionary of arguments to be passed to the residual
                function. The default is None, which corresponds to an empty
                dictionary.
            kwargs: optional keyword arguments
                Additional keyword arguments are sent to the
                lmfit.Minimizer.minimize() method.
        Returns:
            result: lmfit.MinimizerResult
                An object containing the results of the minimization. It is
                also stored in self.emcee_results[label]['result'].
        """
        if label == 'best':
            raise ValueError("'best' is a reserved label and cannot be used")
        elif label not in self.lmfit_results.keys():
            raise ValueError("The MCMC cannot be run unless an lmfit using this label has run first.")
        # set up and do MCMC
        residual_args = (self.z, self.f, *residual_args)
        guess = self.lmfit_results[label]['result'].params
        minimizer = lm.Minimizer(model.residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(method='emcee', **kwargs)
        # get the MLE, median, and 1 sigma uncertainties around the median for each parameter in the flatchain
        one_sigma = 1 - 2 * stats.norm.cdf(-1)
        p = (100 - one_sigma) / 2
        median = {key: np.percentile(result.flatchain[key], 50) for key in result.flatchain.keys()}
        sigma = {key: (np.percentile(result.flatchain[key], p), np.percentile(result.flatchain[key], 100 - p))
                 for key in result.flatchain.keys()}
        mle = dict(result.flatchain.iloc[np.argmax(result.lnprob)])
        # save the results
        self.emcee_results[label] = {'result': result, 'model': model, 'median': median, 'sigma': sigma, 'mle': mle}
        # if the result is better than has been previously computed, add it to the 'best' key
        if 'best' not in self.emcee_results.keys():
            self.emcee_results['best'] = self.lmfit_results[label]
            self.emcee_results['best']['label'] = label
        elif result.aic < self.emcee_results['best']['result'].aic:
            self.emcee_results['best']['result'] = result
            self.emcee_results['best']['objective'] = residual
            self.emcee_results['best']['label'] = label
        return result

    def fit_report(self, label='best', fit_type='lmfit', return_string=False):
        """
        Print a string summarizing a loop fit.
        Args:
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            return_string: boolean
                Return a string with the fit report instead of printing. The
                default is False.

        Returns:
            string: string
                A string containing the fit report. None is output if
                return_string is False.
        """
        _, result_dict = self._get_model(fit_type, label)
        string = lm.fit_report(result_dict['result'])
        if return_string:
            return string
        else:
            print(string)

    def compute_energy_calibration(self, pulse_indices=None, use_mask=True, fix_zero=True, k=2, bc_type='not-a-knot'):
        """
        Compute the response to energy calibration from data in the pulse
        objects. There must be at least two distinct single energy pulses.
        Args:
            pulse_indices: iterable of integers
                Indices of pulse objects in loop.pulses to use for the
                calibration. The default is None and all are used.
            use_mask: boolean
                Determines if the pulse mask is used to filter the pulse
                responses used for the calibration. The default is True.
            fix_zero: boolean
                Determines if the zero point is added as a fixed point in the
                calibration. The default is True.
            k: integer
                The interpolating spline degree. The default is 2.
            bc_type: string or 2-tuple or None
                The type of spline boundary condition. Valid kinds
                correspond to those in scipy.interpolate.make_interp_spline.
                The default is 'not-a-knot'.
        """
        # get energies and responses for the calibration
        responses, energies, _, calibration_type = self._calibration_points(pulse_indices=pulse_indices,
                                                                            use_mask=use_mask, fix_zero=fix_zero)
        assert len(energies) >= 2, "There must be at least 2 pulse data sets with unique, known, single energy lines."
        # sort them by increasing response
        responses, energies = np.array(responses), np.array(energies)
        responses, indices = np.unique(responses, return_index=True)
        energies = energies[indices]
        # store for future plotting
        self._response_avg = responses
        self._response_energies = energies
        self._energy_calibration_type = calibration_type

        self.energy_calibration = make_interp_spline(responses, energies, k=k, bc_type=bc_type)

    def compute_phase_calibration(self, pulse_indices=None, use_mask=True, fix_zero=True, k=2, bc_type='not-a-knot'):
        """
        Compute the energy to phase calibration from data in the pulse objects.
        There must be at least two distinct single energy pulses.
        Args:
            pulse_indices: iterable of integers
                Indices of pulse objects in loop.pulses to use for the
                calibration. The default is None and all are used.
            use_mask: boolean
                Determines if the pulse mask is used to filter the pulse
                responses used for the calibration. The default is True.
            fix_zero: boolean
                Determines if the zero point is added as a fixed point in the
                calibration. The default is True.
            k: integer
                The interpolating spline degree. The default is 2.
            bc_type: string or 2-tuple or None
                The type of spline boundary condition. Valid kinds
                correspond to those in scipy.interpolate.make_interp_spline.
                The default is 'not-a-knot'.
        """
        _, energies, indices, _ = self._calibration_points(pulse_indices=pulse_indices, fix_zero=fix_zero)
        assert len(energies) >= 2, "There must be at least 2 pulse data sets with unique, known, single energy lines."
        # compute phase and amplitude responses
        phase = []
        for pulse in itemgetter(*indices)(self.pulses):
            data = pulse.p_trace[pulse.mask] if use_mask else pulse.p_trace
            phase.append(np.median(pulse.compute_responses("phase_filter", data=data)[0]))
        # sort them by increasing energy
        phase, energies = sort_and_fix(phase, energies, fix_zero)
        # store for future plotting
        self._phase_avg = phase
        self._phase_energies = energies

        self.phase_calibration = make_interp_spline(energies, phase, k=k, bc_type=bc_type)

    def compute_amplitude_calibration(self, pulse_indices=None, use_mask=True, fix_zero=True, k=2,
                                      bc_type='not-a-knot'):
        """
        Compute the energy to amplitude calibration from data in the pulse
        objects. There must be at least two distinct single energy pulses.
        Args:
            pulse_indices: iterable of integers
                Indices of pulse objects in loop.pulses to use for the
                calibration. The default is None and all are used.
            use_mask: boolean
                Determines if the pulse mask is used to filter the pulse
                responses used for the calibration. The default is True.
            fix_zero: boolean
                Determines if the zero point is added as a fixed point in the
                calibration. The default is True.
            k: integer
                The interpolating spline degree. The default is 2.
            bc_type: string or 2-tuple or None
                The type of spline boundary condition. Valid kinds
                correspond to those in scipy.interpolate.make_interp_spline.
                The default is 'not-a-knot'.
        """
        _, energies, indices, _ = self._calibration_points(pulse_indices=pulse_indices, fix_zero=fix_zero)
        assert len(energies) >= 2, "There must be at least 2 pulse data sets with unique, known, single energy lines."
        # compute phase and amplitude responses
        amplitude = []
        for pulse in itemgetter(*indices)(self.pulses):
            data = pulse.a_trace[pulse.mask] if use_mask else pulse.a_trace
            amplitude.append(np.median(pulse.compute_responses("amplitude_filter", data=data)[0]))
        # sort them by increasing energy
        amplitude, energies = sort_and_fix(amplitude, energies, fix_zero)
        # store for future plotting
        self._amplitude_avg = amplitude
        self._amplitude_energies = energies

        self.amplitude_calibration = make_interp_spline(energies, amplitude, k=k, bc_type=bc_type)

    def compute_template_calibration(self, pulse_indices=None, k=1, bc_type='not-a-knot', average_phase=False,
                                     average_amplitude=False):
        """
        Compute the energy to template calibration from data in the pulse
        objects. Each component of the template is normalized to unit height,
        and energies outside of the bounds are extrapolated and may lead to
        poor results.
        Args:
            pulse_indices: iterable of integers
                Indices of pulse objects in loop.pulses to use for the
                calibration. The default is None and all are used.
            k: integer
                The interpolating spline degree. The default is 1.
            bc_type: string or 2-tuple or None
                The type of spline boundary condition. Valid kinds
                correspond to those in scipy.interpolate.make_interp_spline.
                The default is 'not-a-knot'.
            average_phase: boolean
                Average all of the phase templates together to use as the
                template for all energies. The default is False.
            average_amplitude: boolean
                Average all of the amplitude templates together to use as the
                template for all energies. The default is False.
        """
        # find energies and order
        _, energies, indices, _ = self._calibration_points(pulse_indices=pulse_indices, fix_zero=False)
        energies, order = np.unique(energies, return_index=True)
        indices = np.array(indices)[order]

        templates = np.array([pulse.template for pulse in itemgetter(*indices)(self.pulses)])  # energies x 2 x points
        templates /= np.abs(np.min(templates, axis=-1, keepdims=True))  # normalize to unit height on both signals
        if average_phase:
            templates[:, 0, :] = templates[:, 0, :].mean(axis=0, keepdims=True)
        if average_amplitude:
            templates[:, 1, :] = templates[:, 1, :].mean(axis=0, keepdims=True)
        templates_fft = np.fft.rfft(templates, axis=-1)  # energies x 2 x frequencies

        self.template_fft = make_interp_spline(energies, templates_fft, k=k, axis=0, bc_type=bc_type)
        self._template_size = templates.shape[2]

    def template(self, energy):
        """
        An interpolated function that gives the pulse template at all energies
        as computed by pulse.compute_template_calibration().
        Args:
            energy: float
                The energy at which to evaluate the template
        Returns:
            template: numpy.ndarray
                A 2 x N array that represents the phase and amplitude template.
        """
        if self.template_fft is None:
            raise AttributeError("The loop template has not been calculated yet.")
        fft = self.template_fft(energy)
        template = np.fft.irfft(fft, self._template_size)
        return template

    def plot(self, plot_types=("iq", "magnitude", "phase"), plot_fit=False, use_mask=True, label="best",
             fit_type="lmfit", calibrate=False, plot_guess=None, n_rows=2, title=True, title_kwargs=None, legend=True,
             legend_index=0, legend_kwargs=None, fit_parameters=(), parameters_kwargs=None, tighten=True, db=True,
             unwrap=True, plot_kwargs=None, axes_list=None):
        """
        Plot a variety of data representations in a matplotlib pyplot.subplots
        grid.
        Args:
            plot_types: iterable of strings
                The types of plots to show. Valid types are 'iq', 'magnitude',
                and 'phase'. If a fit was computed with the appropriate
                label and fit_type, 'r_iq', 'r_magnitude', and 'r_phase'
                can be used to make residual plots. The default is ('iq',
                'magnitude', 'phase')
            plot_fit: boolean
                Determines whether the fit is plotted or not. The default is
                False. The residual plots can still be rendered if requested in
                the plot_types. fit_parameters and parameters_kwargs are
                ignored if False.
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            calibrate: boolean
                Determines if the plotted data is calibrated by the model. The
                default is False. Residuals aren't calibrated.
            plot_guess: lmfit.Parameters object
                Determines whether the fit guess is plotted or not. The default
                is None.
            n_rows: integer
                An integer specifying how many rows there are in the subplots
                grid. The default is 2.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in figure.suptitle(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True, and a legend is plotted in the first axes. If False,
                legend_kwargs is ignored.
            legend_index: integer
                An integer corresponding to the plot number on which to put the
                legend. The default is 0. It must be less than the number of
                plot_types.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            db: boolean
                Determines if magnitude plots are shown in dB. The default is
                True.
            unwrap: boolean
                Determines if the phase plots are unwrapped. The default is
                True.
            plot_kwargs: an iterable of dictionaries or a single dictionary
                A list of keyword arguments for each plot type. The default is
                None which uses default options. Keywords in this dictionary
                override the default options. Valid keyword arguments can be
                found in the documentation for the specific plot functions.
                'iq': Loop.plot_iq
                'magnitude': Loop.plot_magnitude
                'phase': Loop.plot_phase
                'r_iq': Loop.plot_iq_residual
                'r_magnitude': Loop.plot_magnitude_residual
                'r_phase': Loop.plot_phase_residual
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes on which to put the plots. The default
                is None and a new figure is made.
        Returns:
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes with the plotted data.
        """
        # parse inputs
        if isinstance(plot_types, str):
            plot_types = [plot_types]
        n_plots = len(plot_types)
        n_columns = int(np.ceil(n_plots / n_rows))
        if axes_list is None:
            import matplotlib.pyplot as plt
            figure, axes_list = plt.subplots(n_rows, n_columns, figsize=(4.3 * n_columns, 4.0 * n_rows), squeeze=False)
            axes_list = list(axes_list.flatten())
        else:
            if isinstance(axes_list, np.ndarray):
                axes_list = list(axes_list.flatten())
            figure = axes_list[0].figure
        if plot_kwargs is None:
            plot_kwargs = [{}] * n_plots
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [plot_kwargs] * n_plots
        # make main plots
        index = 0  # if plot_types = ()
        for index, plot_type in enumerate(plot_types):
            kwargs = {"title": False, "legend": False, "axes": axes_list[index], "tighten": tighten,
                      "use_mask": use_mask}
            if plot_type == "iq":
                kwargs.update({"plot_fit": plot_fit, "plot_guess": plot_guess, "label": label, "fit_type": fit_type,
                               "calibrate": calibrate})
                kwargs.update(plot_kwargs[index])
                self.plot_iq(**kwargs)
            elif plot_type == "r_iq":
                kwargs.update(plot_kwargs[index])
                self.plot_iq_residual(**kwargs)
            elif plot_type == "magnitude":
                kwargs.update({"plot_fit": plot_fit, "plot_guess": plot_guess, "label": label, "fit_type": fit_type,
                               "calibrate": calibrate, "db": db})
                kwargs.update(plot_kwargs[index])
                self.plot_magnitude(**kwargs)
            elif plot_type == "r_magnitude":
                kwargs.update(plot_kwargs[index])
                self.plot_magnitude_residual(**kwargs)
            elif plot_type == "phase":
                kwargs.update({"plot_fit": plot_fit, "plot_guess": plot_guess, "label": label, "fit_type": fit_type,
                               "calibrate": calibrate, "unwrap": unwrap})
                kwargs.update(plot_kwargs[index])
                self.plot_phase(**kwargs)
            elif plot_type == "r_phase":
                kwargs.update(plot_kwargs[index])
                self.plot_phase_residual(**kwargs)
            else:
                raise ValueError("'{}' is not a recognized plot type".format(plot_type))
        # turn off unused axes
        if index < len(axes_list) - 1:
            for axes in axes_list[index + 1:]:
                axes.axis('off')
        if plot_fit:
            fit_name, result_dict = self._get_model(fit_type, label)
            if fit_parameters:
                if index == len(axes_list) - 1:
                    axes = axes_list[n_columns - 1]
                    self._make_parameters_textbox(fit_parameters, result_dict['result'], axes, parameters_kwargs)
                else:
                    axes = axes_list[index + 1]
                    text = self._make_parameters_text(fit_parameters, result_dict['result'])
                    kwargs = {"transform": axes.transAxes, "fontsize": 10, "va": "center", "ha": "center",
                              "ma": "center", "bbox": dict(boxstyle='round', facecolor='wheat', alpha=0.5)}
                    if parameters_kwargs is not None:
                        kwargs.update(parameters_kwargs)
                    axes.text(0.5, 0.5, text, **kwargs)
            string = ("power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
                      .format(self.power, self.field, self.temperature * 1000, fit_name))
        else:
            string = ("power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
                      .format(self.power, self.field, self.temperature * 1000))
        if legend:
            kwargs = {}
            if legend_kwargs is not None:
                kwargs.update(legend_kwargs)
            axes_list[legend_index].legend(**kwargs)
        if title:
            title = string if title is True else title
            kwargs = {"fontsize": 11}
            if title_kwargs is not None:
                kwargs.update(title_kwargs)
            if tighten:
                figure.tight_layout()
            figure.suptitle(title, **kwargs).set_y(0.95)
            figure.subplots_adjust(top=0.9)
        elif tighten:
            figure.tight_layout()
        return axes_list

    def plot_iq(self, data_kwargs=None, plot_fit=False, use_mask=True, label="best", fit_type="lmfit", calibrate=False,
                fit_kwargs=None, fit_parameters=(), parameters_kwargs=None, plot_guess=None, guess_kwargs=None,
                x_label=None, y_label=None, label_kwargs=None, legend=True, legend_kwargs=None, title=True,
                title_kwargs=None, tick_kwargs=None, tighten=True, axes=None):
        """
        Plot the IQ data.
        Args:
            data_kwargs: dictionary
                Keyword arguments for the data in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            plot_fit: boolean
                Determines whether the fit is plotted or not. The default is
                False. When False, label, fit_type, fit_kwargs,
                fit_parameters, and parameter_kwargs are ignored.
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            calibrate: boolean
                Determines if the plotted data is calibrated by the model. The
                default is False.
            fit_kwargs: dictionary
                Keyword arguments for the fit in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            plot_guess: lmfit.Parameters object
                Determines whether the fit guess is plotted or not. The default
                is None.
            guess_kwargs: dictionary
                Keyword arguments for the guess in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label. If x_label evaluates to False, parameter_kwargs
                is ignored.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted loop.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "I [V]" if not calibrate else "I",
                             "Q [V]" if not calibrate else "Q", equal=True)
        fd = self.f[self.mask] if use_mask else self.f
        zd = self.z[self.mask] if use_mask else self.z
        # plot data
        kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "data"}
        if data_kwargs is not None:
            kwargs.update(data_kwargs)
        # plot fit
        if plot_fit:
            # get the model
            fit_name, result_dict = self._get_model(fit_type, label)
            if fit_name is None:
                raise ValueError("No fit of type '{}' with the label '{}' has been done".format(fit_type, label))
            result = result_dict['result']
            model = result_dict['model']
            z = zd if not calibrate else model.calibrate(result.params, zd, fd)
            axes.plot(z.real, z.imag, **kwargs)
            # calculate the model values
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, plot_kwargs=fit_kwargs,
                                          use_mask=use_mask, default_kwargs={"linestyle": '--', "label": "fit"})
            axes.plot(m.real, m.imag, **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
            title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
            if fit_parameters:
                self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        else:
            axes.plot(zd.real, zd.imag, **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
            title = string.format(self.power, self.field, self.temperature * 1000) if title is True else title
        # plot guess
        if plot_guess is not None:
            default_kwargs = {"linestyle": '-.', "label": "guess", "color": "k"}
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, params=plot_guess,
                                          use_mask=use_mask, plot_kwargs=guess_kwargs, default_kwargs=default_kwargs)
            axes.plot(m.real, m.imag, **kwargs)
        # finalize the plot
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_iq_residual(self, label="best", fit_type="lmfit", use_mask=True, plot_kwargs=None, fit_parameters=(),
                         parameters_kwargs=None, x_label=None, y_label=None, label_kwargs=None, legend=False,
                         legend_kwargs=None, title=True, title_kwargs=None, tick_kwargs=None, tighten=True, axes=None):
        """
        Plot the residual of the IQ data (data - model).
        Args:
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            plot_kwargs: dictionary
                Keyword arguments for the plot in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label. If x_label evaluates to False, parameter_kwargs
                is ignored.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                False. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted residuals.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "I [V]", "Q [V]", equal=True)
        zd = self.z[self.mask] if use_mask else self.z
        # get the model
        default_kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "residual"}
        f, m, kwargs = get_plot_model(self, fit_type, label, plot_kwargs=plot_kwargs, default_kwargs=default_kwargs,
                                      use_mask=use_mask, n_factor=1)
        axes.plot(zd.real - m.real, zd.imag - m.imag, **kwargs)
        # add fit parameters
        if fit_parameters:
            self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        # finalize the plot
        string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
        title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_magnitude(self, data_kwargs=None, plot_fit=False, use_mask=True, f_scale=1, label="best",
                       fit_type="lmfit", calibrate=False, fit_kwargs=None, fit_parameters=(), parameters_kwargs=None,
                       plot_guess=None, guess_kwargs=None, x_label=None, y_label=None, label_kwargs=None, legend=True,
                       legend_kwargs=None, title=True, title_kwargs=None, tick_kwargs=None, tighten=True, db=True,
                       axes=None):
        """
        Plot the magnitude data.
        Args:
            data_kwargs: dictionary
                Keyword arguments for the data in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            plot_fit: boolean
                Determines whether the fit is plotted or not. The default is
                False. When False, label, fit_type, fit_kwargs,
                fit_parameters, and parameter_kwargs are ignored.
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            f_scale: float
                The frequency scale to use. e.g. 1e3 will plot the frequency in
                kHz. 1 corresponding to Hz is the default.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            calibrate: boolean
                Determines if the plotted data is calibrated by the model. The
                default is False.
            fit_kwargs: dictionary
                Keyword arguments for the fit in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            plot_guess: lmfit.Parameters object
                Determines whether the fit guess is plotted or not. The default
                is None.
            guess_kwargs: dictionary
                Keyword arguments for the guess in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            db: boolean
                Determines if magnitude plots are shown in dB. The default is
                True.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted magnitude.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "frequency [Hz]",
                             "|S₂₁|" if not db else "|S₂₁| [dB]")
        fd = self.f[self.mask] if use_mask else self.f
        zd = self.z[self.mask] if use_mask else self.z
        # plot data
        kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "data"}
        if data_kwargs is not None:
            kwargs.update(data_kwargs)
        # plot fit
        if plot_fit:
            # get the model
            fit_name, result_dict = self._get_model(fit_type, label)
            if fit_name is None:
                raise ValueError("No fit of type '{}' with the label '{}' has been done".format(fit_type, label))
            result = result_dict['result']
            model = result_dict['model']
            z = zd if not calibrate else model.calibrate(result.params, zd, fd)
            axes.plot(fd * 1e9 / f_scale, np.abs(z) if not db else 20 * np.log10(np.abs(z)), **kwargs)
            # calculate the model values
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, plot_kwargs=fit_kwargs,
                                          use_mask=use_mask, default_kwargs={"linestyle": '--', "label": "fit"})
            axes.plot(f * 1e9 / f_scale, np.abs(m) if not db else 20 * np.log10(np.abs(m)), **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
            title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
            if fit_parameters:
                self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        else:
            axes.plot(fd * 1e9 / f_scale, np.abs(zd) if not db else 20 * np.log10(np.abs(zd)), **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
            title = string.format(self.power, self.field, self.temperature * 1000) if title is True else title
        # plot guess
        if plot_guess is not None:
            default_kwargs = {"linestyle": '-.', "label": "guess", "color": "k"}
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, params=plot_guess,
                                          use_mask=use_mask, plot_kwargs=guess_kwargs, default_kwargs=default_kwargs)
            axes.plot(f, np.abs(m) if not db else 20 * np.log10(np.abs(m)), **kwargs)
        # finalize the plot
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_magnitude_residual(self, f_scale=1, label="best", fit_type="lmfit", use_mask=True, plot_kwargs=None,
                                fit_parameters=(), parameters_kwargs=None, x_label=None, y_label=None,
                                label_kwargs=None, legend=False, legend_kwargs=None, title=True, title_kwargs=None,
                                tick_kwargs=None, tighten=True, axes=None):
        """
        Plot the residual of the magnitude data (data - model).
        Args:
            f_scale: float
                The frequency scale to use. e.g. 1e3 will plot the frequency in
                kHz. 1 corresponding to Hz is the default.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            plot_kwargs: dictionary
                Keyword arguments for the plot in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                False. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted residuals.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "frequency [Hz]", "|S₂₁| [V]")
        fd = self.f[self.mask] if use_mask else self.f
        zd = self.z[self.mask] if use_mask else self.z
        # get the model
        default_kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "residual"}
        f, m, kwargs = get_plot_model(self, fit_type, label, plot_kwargs=plot_kwargs, default_kwargs=default_kwargs,
                                      use_mask=use_mask, n_factor=1)
        axes.plot(fd * 1e9 / f_scale, np.abs(zd) - np.abs(m), **kwargs)
        # add fit parameters
        if fit_parameters:
            self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        # finalize the plot
        string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
        title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_phase(self, data_kwargs=None, plot_fit=False, use_mask=True, f_scale=1, label="best", fit_type="lmfit",
                   calibrate=False, fit_kwargs=None, fit_parameters=(), parameters_kwargs=None, plot_guess=None,
                   guess_kwargs=None, x_label=None, y_label=None, label_kwargs=None, legend=True, legend_kwargs=None,
                   title=True, title_kwargs=None, tick_kwargs=None, tighten=True, unwrap=True, axes=None):
        """
        Plot the phase data.
        Args:
            data_kwargs: dictionary
                Keyword arguments for the data in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            plot_fit: boolean
                Determines whether the fit is plotted or not. The default is
                False. When False, label, fit_type, fit_kwargs,
                fit_parameters, and parameter_kwargs are ignored.
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            f_scale: float
                The frequency scale to use. e.g. 1e3 will plot the frequency in
                kHz. 1 corresponding to Hz is the default.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            calibrate: boolean
                Determines if the plotted data is calibrated by the model. The
                default is False.
            fit_kwargs: dictionary
                Keyword arguments for the fit in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            plot_guess: lmfit.Parameters object
                Determines whether the fit guess is plotted or not. The default
                is None.
            guess_kwargs: dictionary
                Keyword arguments for the guess in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            unwrap: boolean
                Determines if the phase is unwrapped or not. The default is
                True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted phase.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "frequency [Hz]", "phase [radians]")
        fd = self.f[self.mask] if use_mask else self.f
        zd = self.z[self.mask] if use_mask else self.z
        # plot data
        kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "data"}
        if data_kwargs is not None:
            kwargs.update(data_kwargs)
        # plot fit
        if plot_fit:
            # get the model
            fit_name, result_dict = self._get_model(fit_type, label)
            if fit_name is None:
                raise ValueError("No fit of type '{}' with the label '{}' has been done".format(fit_type, label))
            result = result_dict['result']
            model = result_dict['model']
            z = zd if not calibrate else model.calibrate(result.params, zd, fd, center=True)
            axes.plot(fd * 1e9 / f_scale, np.unwrap(np.angle(z)) if unwrap else np.angle(z), **kwargs)
            # calculate the model values
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, plot_kwargs=fit_kwargs,
                                          use_mask=use_mask, default_kwargs={"linestyle": '--', "label": "fit"},
                                          center=True)
            offset = 2 * np.pi if np.angle(z[0]) - np.angle(m[0]) > np.pi else 0
            axes.plot(f * 1e9 / f_scale, np.unwrap(np.angle(m) + offset) if unwrap else np.angle(m), **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
            title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
            if fit_parameters:
                self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        else:
            axes.plot(fd * 1e9 / f_scale, np.unwrap(np.angle(zd)) if unwrap else np.angle(zd), **kwargs)
            string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
            title = string.format(self.power, self.field, self.temperature * 1000) if title is True else title
        # plot guess
        if plot_guess is not None:
            default_kwargs = {"linestyle": '-.', "label": "guess", "color": "k"}
            f, m, kwargs = get_plot_model(self, fit_type, label, calibrate=calibrate, params=plot_guess,
                                          use_mask=use_mask, plot_kwargs=guess_kwargs, default_kwargs=default_kwargs)
            axes.plot(f, np.unwrap(np.angle(m)) if unwrap else np.angle(m), **kwargs)
        # finalize the plot
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_phase_residual(self, f_scale=1, label="best", fit_type="lmfit", use_mask=True, plot_kwargs=None,
                            fit_parameters=(), parameters_kwargs=None, x_label=None, y_label=None, label_kwargs=None,
                            legend=False, legend_kwargs=None, title=True, title_kwargs=None, tick_kwargs=None,
                            tighten=True, axes=None):
        """
        Plot the residual of the phase data (data - model).
        Args:
            f_scale: float
                The frequency scale to use. e.g. 1e3 will plot the frequency in
                kHz. 1 corresponding to Hz is the default.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            use_mask: boolean
                Determines whether or not to use the mask for the plotted data.
                The default is True.
            plot_kwargs: dictionary
                Keyword arguments for the plot in axes.plot(). The default is
                None which uses default options. Keywords in this dictionary
                override the default options.
            fit_parameters: iterable of strings
                Parameters to label on the side of the plot. The default is an
                empty tuple corresponding to no labels. 'chi2' can also be
                included in the list to display the reduced chi squared value
                for the fit. If fit_parameters evaluates to False,
                parameter_kwargs is ignored.
            parameters_kwargs: dictionary
                Keyword arguments for the parameters textbox in axes.text().
                The default is None which uses default options. Keywords in
                this dictionary override the default options.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                False. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is True.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted residuals.
        """
        # parse inputs
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, "frequency [Hz]", "phase [radians]")
        fd = self.f[self.mask] if use_mask else self.f
        zd = self.z[self.mask] if use_mask else self.z
        # get the model
        default_kwargs = {"marker": 'o', "markersize": 2, "linestyle": 'None', "label": "residual"}
        f, m, kwargs = get_plot_model(self, fit_type, label, plot_kwargs=plot_kwargs, default_kwargs=default_kwargs,
                                      use_mask=use_mask, n_factor=1)
        axes.plot(fd * 1e9 / f_scale, np.unwrap(np.angle(zd)) - np.unwrap(np.angle(m)), **kwargs)
        # add fit parameters
        if fit_parameters:
            self._make_parameters_textbox(fit_parameters, result, axes, parameters_kwargs)
        # finalize the plot
        string = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
        title = string.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_energy_calibration(self, axes=None):
        """
        Plot the energy calibration.
        Args:
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted calibration.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
        else:
            figure = axes.figure
        xx = np.linspace(np.min(self._response_avg) * 0.8, np.max(self._response_avg) * 1.2, 1000)
        axes.plot(xx, self.energy_calibration(xx), label='calibration')
        axes.plot(self._response_avg, self._response_energies, 'o', label='true')
        axes.set_xlabel('response [radians]')
        axes.set_ylabel('energy [eV]')
        axes.legend()
        figure.tight_layout()
        return axes

    def plot_phase_calibration(self, axes=None):
        """
        Plot the phase calibration.
        Args:
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted calibration.
        """
        figure, axes = setup_axes(axes, 'energy [eV]', 'phase [radians]')
        xx = np.linspace(np.min(self._phase_energies) * 0.8, np.max(self._phase_energies) * 1.2, 1000)
        axes.plot(xx, self.phase_calibration(xx), label='calibration')
        axes.plot(self._phase_energies, self._phase_avg, 'o', label='true')
        finalize_axes(axes, legend=True, tighten=True)
        return axes

    def plot_amplitude_calibration(self, axes=None):
        """
        Plot the phase calibration.
        Args:
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted calibration.
        """
        figure, axes = setup_axes(axes, 'energy [eV]', 'amplitude [radians]')
        xx = np.linspace(np.min(self._amplitude_energies) * 0.8, np.max(self._amplitude_energies) * 1.2, 1000)
        axes.plot(xx, self.amplitude_calibration(xx), label='calibration')
        axes.plot(self._amplitude_energies, self._amplitude_avg, 'o', label='true')
        finalize_axes(axes, legend=True, tighten=True)
        return axes

    def plot_spectra(self, pulse_indices=None, plot_kwargs=None, hist_kwargs=None, x_limits=None, second_x_axis=False,
                     x_label=None, y_label=None, label_kwargs=None, legend=True, legend_kwargs=None, tick_kwargs=None,
                     tighten=True, norm=False, axes=None):
        """
        Plot the spectrum of the pulse responses in the loop.
        Args:
            pulse_indices: iterable of integers (optional)
                Indices of pulse objects in loop.pulses to include in the total
                spectrum. The default is None and all pulses are used.
            plot_kwargs: dictionary or list of dictionaries
                Keyword arguments for axes.plot(). The default is None which
                uses the default options. Keywords in this dictionary override
                the default options. If a list of dictionaries is given, the
                order corresponds to pulse_indices.
            hist_kwargs: dictionary
                Keyword arguments for axes.hist(). The default is None which
                uses the default options. Keywords in this dictionary override
                the default options.
            x_limits: length 2 iterable of floats
                Bounds on the x-axis
            second_x_axis: boolean (optional)
                If True, a second x-axis is plotted below the first with the
                wavelength values. The default is False.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label. If x_label evaluates to False, parameter_kwargs
                is ignored.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            norm: boolean
                If False, the combined spectra is normalized to unit
                probability where the relative brightness of the sub-spectra
                is retained. If True, each sub-spectra is normalized to the
                same probability and the total probability is still 1. The
                default is False.
            axes: matplotlib.axes.Axes class (optional)
                An axes class for plotting the data.
        Returns:
            axes: matplotlib.axes.Axes class
                An axes class with the plotted data.
        """
        # mask the pulses
        if pulse_indices is None:
            pulses = self.pulses
        else:
            try:
                pulses = itemgetter(*pulse_indices)(self.pulses)
            except TypeError:
                pulses = itemgetter(pulse_indices)(self.pulses)
            if not isinstance(pulses, tuple):
                pulses = [pulses]
        # get all of the energies
        energies = []
        norms = []
        min_bandwidth = np.inf
        for pulse in pulses:
            try:
                bandwidth = pulse.spectrum["bandwidth"]
                norms.append(pulse.spectrum["energies"].size)
                energies.append(pulse.spectrum["energies"])
                min_bandwidth = bandwidth if bandwidth < min_bandwidth else min_bandwidth
            except AttributeError:
                pass
        energies = np.concatenate(energies)
        max_energy, min_energy = energies.max(), energies.min()

        # check if calibrated
        calibrated = pulses[0].spectrum["calibrated"]

        # setup axes and plot data
        figure, axes = setup_axes(axes, x_label, y_label, label_kwargs=label_kwargs,
                                  x_label_default='energy [eV]' if calibrated else "response [radians]",
                                  y_label_default='probability density')
        kwargs = {"bins": 10 * int((max_energy - min_energy) / min_bandwidth), "density": True}
        if norm is True:
            prob_norm = 0
            weight = 0
            for p in pulses:
                prob_norm += p.spectrum['pdf'](energies)
            for p in pulses:
                weight += p.spectrum["pdf"](energies) / p.spectrum["energies"].size / prob_norm
            kwargs.update({"weights": weight})
        if hist_kwargs is not None:
            kwargs.update(hist_kwargs)
        axes.hist(energies, **kwargs)

        # plot the PDFs
        label = ""
        if plot_kwargs is None:
            plot_kwargs = {}
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [plot_kwargs] * len(pulses)
        for index, pulse in enumerate(pulses):
            # get the needed data from the spectrum dictionary
            pdf = pulse.spectrum["pdf"]
            bandwidth = pulse.spectrum["bandwidth"]
            # plot the data
            n_bins = 10 * int((max_energy - min_energy) / bandwidth)
            xx = np.linspace(min_energy, max_energy,  10 * n_bins)
            if not np.isnan(pulse.resolving_power):
                label = "{:.0f} nm: R = {:.2f}".format(ev_nm_convert(pulse.energies[0]), pulse.resolving_power)
            else:
                label = ""
            kwargs = {"label": label}
            kwargs.update(plot_kwargs[index])
            pdf_xx = pdf(xx) / len(norms) if norm else norms[index] * pdf(xx) / np.sum(norms)
            axes.plot(xx, pdf_xx, **kwargs)

        # set x axis limits
        if x_limits is not None:
            axes.set_xlim(x_limits)
        else:
            axes.set_xlim([min_energy, max_energy])
        # have to set tick parameters before making next axis or tick labels get messed up
        if tick_kwargs is not None:
            axes.tick_params(**tick_kwargs)
        # put twin axis on the bottom
        if second_x_axis:
            wvl_axes = axes.twiny()
            wvl_axes.set_frame_on(True)
            wvl_axes.patch.set_visible(False)
            wvl_axes.xaxis.set_ticks_position('bottom')
            wvl_axes.xaxis.set_label_position('bottom')
            wvl_axes.spines['bottom'].set_position(('outward', 40))
            kwargs = {}
            if label_kwargs is not None:
                kwargs.update(label_kwargs)
            wvl_axes.set_xlabel('wavelength [nm]', **kwargs)
            if x_limits is not None:
                wvl_axes.set_xlim(x_limits)
            else:
                wvl_axes.set_xlim([min_energy, max_energy])
            if tick_kwargs is not None:
                wvl_axes.tick_params(**tick_kwargs)

            # redo ticks on bottom axis
            def tick_labels(x):
                v = ev_nm_convert(x)
                return ["%.0f" % z for z in v]

            # set wavelength ticks
            x_locs = axes.xaxis.get_majorticklocs()
            wvl_axes.set_xticks(x_locs)
            wvl_axes.set_xticklabels(tick_labels(x_locs))

        finalize_axes(axes, legend=legend and label, legend_kwargs=legend_kwargs, tighten=tighten)
        return axes

    def _calibration_points(self, pulse_indices=None, use_mask=True, fix_zero=True):
        skip_bad = True if pulse_indices is None else False
        pulse_indices = range(len(self.pulses)) if pulse_indices is None else pulse_indices
        responses = []
        energies = []
        indices = list(pulse_indices)  # creates a copy
        bad_indices = []
        calibration_type = None
        # get energies and responses for pulses
        for index, pulse in enumerate(itemgetter(*pulse_indices)(self.pulses)):
            if calibration_type is None:
                calibration_type = pulse._response_type
            elif calibration_type != pulse._response_type:
                raise ValueError("pulse objects have different types of responses")
            energy = pulse.energies
            # skip if no energy
            if energy is None and skip_bad:
                bad_indices.append(index)
                continue
            try:
                n_energy = len(energy)
            except TypeError:
                n_energy = 1
                energy = [energy]
            # skip if more than one energy
            if n_energy != 1 and skip_bad:
                bad_indices.append(index)
                continue
            # record results
            assert n_energy == 1, "only pulses with single energies can be used as calibration points"
            energies.append(energy[0])
            responses.append(np.median(pulse.responses[pulse.mask]) if use_mask else np.median(pulse.responses))
        # remove indices we skipped from the index list
        for index in sorted(bad_indices, reverse=True):
            del indices[index]
        # add zero point
        if fix_zero:
            responses, energies = [0] + responses, [0] + energies
        return responses, energies, indices, calibration_type

    def _set_directory(self, directory):
        self._directory = directory
        for noise in self.noise:
            noise._set_directory(self._directory)
        for pulse in self.pulses:
            pulse._set_directory(self._directory)

    def _get_model(self, fit_type, label):
        if fit_type not in ['lmfit', 'emcee', 'emcee_mle']:
            raise ValueError("'fit_type' must be either 'lmfit', 'emcee', or 'emcee_mle'")
        if fit_type == "lmfit" and label in self.lmfit_results.keys():
            result_dict = self.lmfit_results[label]
            original_label = self.lmfit_results[label]["label"] if label == "best" else label
        elif fit_type == "emcee" and label in self.emcee_results.keys():
            result_dict = self.emcee_results[label]
            original_label = self.lmfit_results[label]["label"] if label == "best" else label
        elif fit_type == "emcee_mle" and label in self.emcee_results.keys():
            result_dict = copy.deepcopy(self.emcee_results[label])
            for name in result_dict['result'].params.keys():
                result_dict['result'].params[name].set(value=self.emcee_results[label]["mle"][name])
            original_label = self.lmfit_results[label]["label"] if label == "best" else label
        else:
            result_dict = None
            original_label = None
        return original_label, result_dict

    def _make_parameters_textbox(self, fit_parameters, result, axes, parameters_kwargs):
        text = self._make_parameters_text(fit_parameters, result)
        kwargs = {"transform": axes.transAxes, "fontsize": 10, "va": "top", "ha": "left", "ma": "center",
                  "bbox": dict(boxstyle='round', facecolor='wheat', alpha=0.5)}
        if parameters_kwargs is not None:
            kwargs.update(parameters_kwargs)
        axes_width = axes.bbox.width
        t = axes.text(1.05, 0.95, text, **kwargs)
        if t.get_bbox_patch() is not None:
            text_width = t.get_bbox_patch().get_width()
        else:
            text_width = 0.1 * axes_width
        axes.figure.set_figwidth(axes.figure.get_figwidth() + text_width)

    @staticmethod
    def _make_parameters_text(fit_parameters, result):
        text = []
        for name in fit_parameters:
            if name == "chi2":
                text.append(r"$\chi^2 = {:g}$".format(result.redchi))
            else:
                text.append("{} = {:g}".format(name, result.params[name].value))
        text = "\n".join(text)

        return text
