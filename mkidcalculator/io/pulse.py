import os
import corner
import pickle
import logging
import warnings
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider

from mkidcalculator.io.data import AnalogReadoutPulse
from mkidcalculator.io.utils import (compute_phase_and_amplitude, offload_data, _loaded_npz_files,
                                     quadratic_spline_roots, ev_nm_convert)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Pulse:
    """A class for manipulating the pulse data."""
    def __init__(self):
        # pulse data
        self._data = AnalogReadoutPulse()  # dummy class replaced by load()
        # loop reference for computing phase and amplitude
        self._loop = None
        # noise reference for computing energies
        self._noise = None
        # phase and amplitude data
        self._p_trace = None
        self._a_trace = None
        # template attributes
        self._traces = None
        self._template = None
        # filter attributes
        self._p_trace_filtered = None
        self._optimal_filter = None
        self._optimal_filter_var = None
        self._p_filter = None
        self._p_filter_var = None
        self._a_filter = None
        self._a_filter_var = None
        # detector response
        self._responses = None
        self._peak_indices = None
        # trace mask
        self._mask = None
        self._prepulse_mean = None
        self._prepulse_rms = None
        self._postpulse_min_slope = None
        # for holding large data
        self._npz = None
        self._directory = None
        # response information
        self._spectrum = None

        log.info("Pulse object created. ID: {}".format(id(self)))

    def __getstate__(self):
        return offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="pulse_data_")

    @property
    def f_bias(self):
        """The bias frequency for the data set."""
        return self._data["f_bias"]

    @property
    def i_trace(self):
        """The mixer I output traces."""
        return self._data["i_trace"]

    @property
    def q_trace(self):
        """The mixer Q output traces."""
        return self._data["q_trace"]

    @property
    def offset(self):
        """The mixer IQ offset at the bias frequency."""
        return self._data["offset"]

    @property
    def metadata(self):
        """A dictionary containing metadata about the pulse."""
        return self._data["metadata"]

    @property
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def energies(self):
        """The known photon energies in this data set."""
        return self._data["energies"]

    @property
    def sample_rate(self):
        """The sample rate of the IQ data."""
        return self._data['sample_rate']

    @property
    def p_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        pulse.compute_phase_and_amplitude() is run.
        """
        if self._p_trace is None:
            raise AttributeError("The phase information has not been computed yet.")
        if isinstance(self._p_trace, str):
            return _loaded_npz_files[self._npz][self._p_trace]
        else:
            return self._p_trace

    @p_trace.setter
    def p_trace(self, phase_trace):
        self._p_trace = phase_trace

    @property
    def a_trace(self):
        """
        A settable property that contains the amplitude trace information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        pulse.compute_phase_and_amplitude() is run.
        """
        if self._a_trace is None:
            raise AttributeError("The amplitude information has not been computed yet.")
        if isinstance(self._a_trace, str):
            return _loaded_npz_files[self._npz][self._a_trace]
        else:
            return self._a_trace

    @a_trace.setter
    def a_trace(self, amplitude_trace):
        self._a_trace = amplitude_trace

    @property
    def loop(self):
        """
        A settable property that contains the Loop object required for doing
        pulse calculations like computing the phase and amplitude traces. If
        the loop has not been set, it will raise an AttributeError. When the
        loop is set, all information created from the previous loop is deleted.
        """
        if self._loop is None:
            raise AttributeError("The loop object for this pulse has not been set yet.")
        return self._loop

    @loop.setter
    def loop(self, loop):
        self._loop = loop
        self.clear_loop_data()
        try:
            self.noise.loop = self.loop
        except AttributeError:
            pass

    @property
    def noise(self):
        """
        A settable property that contains the Noise object required for doing
        pulse calculations like optimal filtering. If the noise has not been
        set, it will raise an AttributeError. When the noise is set, all
        information created from the previous noise is deleted.
        """
        if self._noise is None:
            raise AttributeError("The noise object for this pulse has not been set yet.")
        return self._noise

    @noise.setter
    def noise(self, noise):
        if self._noise is not noise:
            self.clear_noise_data()
        self._noise = noise
        try:
            self.noise.loop = self.loop
        except AttributeError:
            pass

    @property
    def responses(self):
        """
        A settable property that contains the detector response amplitudes made
        with pulse.compute_responses().
        """
        if self._responses is None:
            raise AttributeError("The responses for this pulse have not been calculated yet.")
        return self._responses

    @responses.setter
    def responses(self, responses):
        self.clear_responses()
        self._responses = responses

    @property
    def peak_indices(self):
        """
        A settable property that contains the indices of the trace pulse
        arrivals made with pulse.compute_responses().
        """
        if self._peak_indices is None:
            raise AttributeError("The peak indices for this pulse have not been calculated yet.")
        return self._peak_indices

    @peak_indices.setter
    def peak_indices(self, peak_indices):
        self.clear_peak_indices()
        self._peak_indices = peak_indices

    @property
    def template(self):
        """
        A settable property that contains the phase and amplitude templates
        made with pulse.make_template().
        """
        if self._template is None:
            raise AttributeError("The template for this pulse has not been calculated yet.")
        return self._template

    @template.setter
    def template(self, template):
        self.clear_template()
        self._template = template

    @property
    def optimal_filter(self):
        """
        A property that contains the optimal filter made with
        pulse.make_filters().
        """
        if self._optimal_filter is None:
            raise AttributeError("The optimal filter for this pulse has not been calculated yet.")
        return self._optimal_filter

    @property
    def optimal_filter_var(self):
        """
        A property that contains the optimal filter expected variance made with
        pulse.make_filters().
        """
        if self._optimal_filter_var is None:
            raise AttributeError("The optimal filter for this pulse has not been calculated yet.")
        return self._optimal_filter_var

    @property
    def p_filter(self):
        """
        A property that contains the phase filter made with
        pulse.make_filters().
        """
        if self._p_filter is None:
            raise AttributeError("The phase filter for this pulse has not been calculated yet.")
        return self._p_filter

    @property
    def p_filter_var(self):
        """
        A property that contains the phase filter expected variance made with
        pulse.make_filters().
        """
        if self._p_filter_var is None:
            raise AttributeError("The phase filter for this pulse has not been calculated yet.")
        return self._p_filter_var

    @property
    def a_filter(self):
        """
        A property that contains the amplitude filter made with
        pulse.make_filters().
        """
        if self._a_filter is None:
            raise AttributeError("The amplitude filter for this pulse has not been calculated yet.")
        return self._a_filter

    @property
    def a_filter_var(self):
        """
        A property that contains the phase filter expected variance made with
        pulse.make_filters().
        """
        if self._a_filter_var is None:
            raise AttributeError("The amplitude filter for this pulse has not been calculated yet.")
        return self._a_filter_var

    @property
    def mask(self):
        """
        A settable property that contains a boolean array that can select trace
        indices from pulse.responses, pulse.i_trace, pulse.q_trace,
        pulse.p_trace, or pulse.a_trace.
        """
        if self._mask is None:
            self._mask = np.ones(self.i_trace.shape[0], dtype=bool)
        return self._mask

    @mask.setter
    def mask(self, mask):
        self.clear_mask()
        self._mask = mask

    @property
    def spectrum(self):
        """
        A dictionary that returns information about the spectrum.
        Keys:
            pdf: scipy.stats.gaussian_kde
                A function that evaluates the pdf of the spectrum.
            interpolation: scipy.interpolate.InterpolatedUnivariateSpline
                An interpolation function that approximates pdf which can
                be easily manipulated to compute derivatives and roots.
            energies: numpy.ndarray
                The energies (or responses--see calibrated) used to calculate
                the spectrum.
            calibrated: boolean
                A boolean describing if the spectrum is calibrated. If it is
                False, the energies correspond to detector responses.
            bandwidth: float
                The kernel bandwidth used for the kernel density estimation.
            peak: float or numpy .nan
                The peak energy of the maximum of the distribution. If there
                is no peak (highly unlikely) than it is set to numpy.nan
            fwhm: float or numpy.nan
                The full width half max of the distribution if it can be
                calculated. If it can't, it is set to numpy.nan
        """
        if self._spectrum is None:
            raise AttributeError("The spectrum for this pulse has not been computed yet.")
        return self._spectrum

    @property
    def resolving_power(self):
        """
        The resolving power for the data set. The spectrum must be computed
        first. If the spectrum is calibrated and pulse.energies has length 1,
        pulse.energies[0] is used as the energy. Otherwise, the mode of the
        spectrum is used.
        """
        peak = self.energies[0] if len(self.energies) == 1 and self.spectrum["calibrated"] else self.spectrum["peak"]
        fwhm = self.spectrum["fwhm"]
        return peak / fwhm

    def clear_loop_data(self):
        """Remove all data calculated from the pulse.loop attribute."""
        self.clear_traces()
        self.clear_mask()

    def clear_traces(self):
        """
        Remove all trace data calculated from pulse.i_trace and pulse.q_trace.
        """
        self._a_trace = None
        self._p_trace = None
        self._npz = None
        self._postpulse_min_slope = None
        self._prepulse_rms = None
        self._prepulse_mean = None
        self.clear_template()
        self.free_memory()

    def clear_noise_data(self):
        """Remove all data calculated from the pulse.noise attribute."""
        self.clear_filters()

    def clear_template(self):
        """
        Clear the template made with pulse.make_template() and reset the
        traces used to make the template.
        """
        self._template = None
        self.clear_filters()

    def clear_filters(self):
        """
        Clear the filters made with pulse.make_filters() and data made with them.
        """
        self._p_trace_filtered = None
        self._optimal_filter = None
        self._optimal_filter_var = None
        self._p_filter = None
        self._p_filter_var = None
        self._a_filter = None
        self._a_filter_var = None
        self.clear_responses()
        self.clear_peak_indices()

    def clear_mask(self):
        """
        Clear all mask data.
        """
        self._mask = None

    def clear_responses(self):
        """
        Clear the response data and any mask information associated with it.
        """
        self._responses = None
        self.clear_spectrum()

    def clear_peak_indices(self):
        """
        Clear the peak indices and any mask information associated with it.
        """
        self._peak_indices = None
        self.clear_spectrum()

    def clear_spectrum(self):
        """
        Clear the spectrum information.
        """
        self._spectrum = None

    def free_memory(self, directory=None, noise=True):
        """
        Offloads a_traces and p_traces to an npz file if they haven't been
        offloaded already and removes any npz file objects from memory, keeping
        just the file name. It doesn't do anything if they don't exist.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
            noise: bool
                If true, the pulse.noise.free_memory() method is called as
                well. If pulse.noise doesn't exist, nothing happens. The
                default is True.
        """
        if directory is not None:
            self._set_directory(directory)
        offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="pulse_data_")
        if isinstance(self._npz, str):  # there might not be an npz file yet
            _loaded_npz_files.free_memory(self._npz)
        try:
            self._data.free_memory()
        except AttributeError:
            pass
        if noise:
            try:
                self.noise.free_memory()
            except AttributeError:
                pass

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", fr=None, center=None, unwrap=True,
                                    noise=False):
        """
        Compute the phase and amplitude traces stored in pulse.p_trace and
        pulse.a_trace.
        Args:
            label: string
                Corresponds to the label in the loop.lmfit_results or
                loop.emcee_results dictionaries where the fit parameters are.
                The resulting DataFrame is stored in
                self.loop_parameters[label]. The default is "best", which gets
                the parameters from the best fits.
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            fr: string
                The parameter name that corresponds to the resonance frequency.
                The default is None which gives the resonance frequency for the
                mkidcalculator.S21 model. This parameter determines the zero
                point for the traces.
            center: string
                An expression of parameters corresponding to the calibrated
                loop center. The default is None which gives the loop center
                for the mkidcalculator.S21 model.
            unwrap: boolean
                Determines whether or not to unwrap the phase data. The default
                is True.
            noise: boolean
                Determines whether or not to also compute the phase and
                amplitude for the noise. The default is false.
        """
        compute_phase_and_amplitude(self, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)
        if noise:
            compute_phase_and_amplitude(self.noise, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Pulse class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            pulse = pickle.load(f)
        assert isinstance(pulse, cls), "'{}' does not contain a Pulse class.".format(file_name)
        return pulse

    @classmethod
    def load(cls, pulse_file_name, data=AnalogReadoutPulse, loop=None, noise=None, **kwargs):
        """
        Pulse class factory method that returns a Pulse() with the data loaded.
        Args:
            pulse_file_name: string
                The file name for the pulse data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the AnalogReadoutPulse class, which interfaces
                with the data products from the analogreadout module.
            loop: Loop object (optional)
                The Loop object needed for computing phase and amplitude. It
                can be specified later or changed with pulse.loop = loop. The
                default is None, which signifies that the loop has not been
                set.
            noise: Noise object (optional)
                The Noise object needed for computing the pulse energies. It
                can be specified later or changed with pulse.noise = noise. The
                default is None, which signifies that the noise has not been
                set.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            pulse: object
                A Pulse() object containing the loaded data.
        """
        pulse = cls()
        pulse._data = data(pulse_file_name, **kwargs)
        if loop is not None:  # don't set loop unless needed.
            pulse.loop = loop
        if noise is not None:
            pulse.noise = noise
        return pulse

    def make_template(self, use_mask=False):
        """
        Make a template from phase and amplitude data. The template is needed
        for computing a filter.
        Args:
            use_mask: bool
                A boolean that determines if the pulse.mask is used as an
                initial filter on the data before creating the template.
        """
        # create a rough template by cutting the noise traces and averaging
        if use_mask:
            self._traces = self._remove_baseline(np.array([self.p_trace[self.mask, :], self.a_trace[self.mask, :]]))
        else:
            self._traces = self._remove_baseline(np.array([self.p_trace, self.a_trace]))
        self._threshold_cut()
        self._average_pulses()
        # make a filter with the template
        self.make_filters()
        # do a better job using a filter
        if use_mask:
            self._traces = self._remove_baseline(np.array([self.p_trace[self.mask, :], self.a_trace[self.mask, :]]))
        else:
            self._traces = self._remove_baseline(np.array([self.p_trace, self.a_trace]))
        self._p_trace_filtered = self.apply_filter(self._traces[0], filter_type="phase_filter")
        self._threshold_cut(use_filter=True)
        self._offset_correction()
        self._average_pulses()
        self._traces = None  # release the memory since we no longer need this

    def make_filters(self):
        """
        Make an optimal filter assuming a linear response and stationary noise.
        A full 2D filter is made as well as two 1D filters for the phase and
        amplitude responses.
        """
        self.clear_filters()
        # pull out some parameters
        n_samples = len(self.template[0])
        sample_rate = self.sample_rate
        shape = (2, n_samples // 2 + 1)
        if self.noise.pp_psd.size != shape[1]:
            raise ValueError("The noise data PSDs must have a shape compatible with the pulse data")
        # assemble noise matrix
        s = np.array([[self.noise.pp_psd, self.noise.pa_psd],
                      [np.conj(self.noise.pa_psd), self.noise.aa_psd]], dtype=np.complex)

        # normalize the template for response = phase + amplitude
        template = self.template / np.abs(self.template[0].min() + self.template[1].min())
        template_fft = fft.rfft(template)
        # compute the optimal filter: conj(template_fft) @ s_inv (single sided)
        filter_fft = np.zeros(shape, dtype=np.complex)
        for index in range(1, shape[1]):
            filter_fft[:, index] = la.lstsq(s[:, :, index].T, np.conj(template_fft[:, index]), rcond=None)[0]
        # return to time domain
        self._optimal_filter = fft.irfft(filter_fft, n_samples)
        # compute the variance with the un-normalized filter
        f_fft = filter_fft[..., np.newaxis].transpose(1, 2, 0)
        t_fft = template_fft[..., np.newaxis].transpose(1, 0, 2)
        self._optimal_filter_var = (sample_rate * n_samples / (4 * np.sum(f_fft @ t_fft).real))
        # normalize the optimal filter to unit response on the template
        norm = self.apply_filter(template, filter_type="optimal_filter").max()
        self._optimal_filter /= norm

        # normalize the template for response = phase
        template = self.template[0, :] / np.abs(self.template[0].min())
        template_fft = fft.rfft(template)
        # compute the phase only optimal filter: conj(phase_fft) / s (single sided)
        phase_filter_fft = np.conj(template_fft) / self.noise.pp_psd
        phase_filter_fft[0] = 0  # discard zero bin for AC coupled filter
        self._p_filter = fft.irfft(phase_filter_fft, n_samples)
        # compute the variance with the un-normalized filter
        self._p_filter_var = (sample_rate * n_samples / (4 * (phase_filter_fft @ template_fft).real))
        # normalize
        norm = self.apply_filter(template, filter_type="phase_filter").max()
        self._p_filter /= norm

        # normalize the template for response = amplitude
        template = self.template[1, :] / np.abs(self.template[1].min())
        template_fft = fft.rfft(template)
        # compute the amplitude only optimal filter: conj(amplitude_fft) / s (single sided)
        amplitude_filter_fft = np.conj(template_fft) / self.noise.aa_psd
        amplitude_filter_fft[0] = 0  # discard zero bin for AC coupled filter
        self._a_filter = fft.irfft(amplitude_filter_fft, n_samples)
        # compute the variance with the un-normalized filter
        self._a_filter_var = (sample_rate * n_samples / (4 * (amplitude_filter_fft @ template_fft).real))
        # normalize
        norm = self.apply_filter(template, filter_type="amplitude_filter").max()
        self._a_filter /= norm

    def performance(self, calculation_type="optimal_filter", mode="variance", response=1.5, baseline=(1, .1),
                    distribution=False):
        """
        Return the expected variance for a particular response calculation
        type.
        Args:
            calculation_type: string
                Valid options are listed. The default is "optimal_filter".
                Options that are Monte Carlo simulations are computationally
                expensive and re-run on each call of pulse.variance().
                "optimal_filter"
                    Theoretical two dimensional optimal filter performance.
                "optimal_filter_mc"
                    Monte Carlo simulation of the two dimensional optimal
                    filter performance.
                "phase_filter"
                    Theoretical phase optimal filter performance.
                "phase_filter_mc"
                    Monte Carlo simulation of the phase optimal filter
                    performance.
                "amplitude_filter"
                    Theoretical amplitude optimal filter performance.
                "amplitude_filter_mc"
                    Monte Carlo simulation of the amplitude optimal filter
                    performance.
            mode: string
                Valid options are listed below the default is "variance".
                "variance"
                    The variance in the response estimation. The returned value
                    should be independent of the response and baseline keywords
                    for the non-Monte Carlo methods.
                "fwhm"
                    The full width half max of the response estimation. The
                    returned value should  be independent of the response and
                    baseline keywords for the non-Monte Carlo methods. A
                    gaussian relationship between the variance and fwhm is
                    assumed for non-Monte Carlo methods.
                "resolving_power"
                    The resolving power of the response estimation defined as
                    the mean response / the resolution. The mean response may
                    differ from the requested response due to bias in the
                    calculation type if a Monte Carlo method is being used.
            response: float
                The combined phase and amplitude response in radians to use in
                the calculation. This is only used in Monte Carlo methods and
                for computing the resolving power. The default is 1.
            baseline: float or iterable of two floats
                The baseline of the photon response. This is only used in Monte
                Carlo methods. If a tuple, it corresponds to (phase baseline,
                amplitude baseline). Otherwise, both are assumed to be the
                same. The default is (1, .1).
            distribution: bool
                Determines if the detector responses are returned as well. This
                is only allowed if the calculation was a Monte Carlo method.
        Returns:
            result: float
                Either the variance, resolution, or resolving power of the
                calculation type.
            responses: numpy.ndarray
                The responses used to calculate the result. This is only
                available for Monte Carlo calculation types and if
                distribution is True.
        """
        # check inputs
        if mode not in ["variance", "fwhm", "resolving_power"]:
            raise ValueError("'{}' is not a valid mode argument".format(mode))
        baseline = np.atleast_1d(baseline)
        if baseline.size == 1:
            baseline = np.ones(2) * baseline
        elif baseline.size != 2:
            raise ValueError("The baseline keyword must be a number or have length 2.")
        # theory calculations
        if calculation_type in ["optimal_filter", "phase_filter", "amplitude_filter"]:
            if calculation_type == "optimal_filter":
                result = self.optimal_filter_var
            elif calculation_type == "phase_filter":
                result = self.p_filter_var
                response *= np.abs(self.template[0].min() / (self.template[0].min() + self.template[1].min()))
            else:
                result = self.a_filter_var
                response *= np.abs(self.template[1].min() / (self.template[0].min() + self.template[1].min()))
            if mode == 'fwhm':
                result = 2 * np.sqrt(2 * np.log(2) * result)
            elif mode == 'resolving_power':
                result = response / (2 * np.sqrt(2 * np.log(2) * result))
            responses = None
        # Monte Carlo calculations
        elif calculation_type in ["optimal_filter_mc", "phase_filter_mc", "amplitude_filter_mc"]:
            if calculation_type == "optimal_filter_mc":
                # get noise traces
                noise = self.noise.generate_noise(noise_type="pa", n_traces=10000)
                # normalize the template for response = phase + amplitude
                template = self.template / np.abs(self.template[0].min() + self.template[1].min())
                # make data traces
                data = noise + response * template[:, np.newaxis, :] + baseline[:, np.newaxis, np.newaxis]
                # compute the responses
                responses, _ = self.compute_responses(calculation_type="optimal_filter", data=data)
            elif calculation_type == "phase_filter_mc":
                # get noise traces
                noise = self.noise.generate_noise(noise_type="p", n_traces=10000)
                # normalize the template for response = phase + amplitude
                template = self.template[0] / np.abs(self.template[0].min() + self.template[1].min())
                # make data traces
                data = noise + response * template + baseline[0]
                # compute the responses
                responses, _ = self.compute_responses(calculation_type="phase_filter", data=data)
            else:
                # get noise traces
                noise = self.noise.generate_noise(noise_type="a", n_traces=10000)
                # normalize the template for response = phase + amplitude
                template = self.template[1] / np.abs(self.template[0].min() + self.template[1].min())
                data = noise + response * template + baseline[1]
                # compute the responses
                responses, _ = self.compute_responses(calculation_type="amplitude_filter", data=data)
            # compute the result
            if mode == 'variance':
                result = np.var(responses, ddof=1)
            elif mode == 'fwhm':
                result, _, _, _, = self._compute_fwhm(responses)
            else:
                fwhm, _, _, _, = self._compute_fwhm(responses)
                result = np.mean(responses) / fwhm
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(calculation_type))
        # return results
        if distribution:
            if responses is None:
                raise ValueError("The distribution can not be returned for non-Monte Carlo methods.")
            return result, responses
        else:
            return result

    def compute_responses(self, calculation_type="optimal_filter", data=None):
        """
        Compute the detector response responses and peak indices using a
        particular calculation method. The results are stored in
        pulse.responses and pulse.peak_indices if data is not None.
        Args:
            calculation_type: string
                The calculation type used to compute the responses. Valid
                options are listed below. The default is "optimal_filter".
                "optimal_filter"
                "phase_filter"
                "amplitude_filter"
            data: numpy.ndarray
                A numpy array for which to compute responses. The data must
                be in the shape specified by pulse.apply_filter for the given
                calculation type. The responses and peak indices are returned
                instead of saved to the object if data is not None. None is the
                default.
        Returns:
             responses: numpy.ndarray
                The response in radians for each trace.
             peak_indices: numpy.ndarray
                The response arrival time index for each trace.
        """
        save_values = data is None
        if calculation_type == "optimal_filter":
            if data is None:
                data = np.array([self.p_trace, self.a_trace])
            filtered_data = self.apply_filter(data, filter_type=calculation_type)
            responses = filtered_data.max(axis=1)
            peak_indices = np.argmax(filtered_data, axis=1)
        elif calculation_type == "phase_filter":
            if data is None:
                data = self.p_trace
            filtered_data = self.apply_filter(data, filter_type=calculation_type)
            responses = filtered_data.max(axis=1)
            peak_indices = np.argmax(filtered_data, axis=1)
        elif calculation_type == "amplitude_filter":
            if data is None:
                data = self.a_trace
            filtered_data = self.apply_filter(data, filter_type=calculation_type)
            responses = filtered_data.max(axis=1)
            peak_indices = np.argmax(filtered_data, axis=1)
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(calculation_type))
        if save_values:
            self.responses = responses
            self.peak_indices = peak_indices
        return responses, peak_indices

    def apply_filter(self, data, filter_type="optimal_filter"):
        """
        Method for convolving the filters with the data. For the 2D filter the
        first axis must be for the phase and amplitude. The filter is applied
        to the last axis.
        """
        size = data.shape[-1]
        pad_back = size // 2
        pad_front = size - pad_back - 1
        pad = [(0, 0)] * (data.ndim - 1)
        pad.append((pad_front, pad_back))
        padded_data = np.pad(data, pad, 'wrap')
        kwargs = {"mode": "valid", "axes": -1}
        if filter_type == "optimal_filter":
            result = (fftconvolve(np.atleast_2d(padded_data[0]), np.atleast_2d(self.optimal_filter[0]), **kwargs) +
                      fftconvolve(np.atleast_2d(padded_data[1]), np.atleast_2d(self.optimal_filter[1]), **kwargs))
        elif filter_type == "phase_filter":
            result = fftconvolve(np.atleast_2d(padded_data), np.atleast_2d(self.p_filter), **kwargs)
        elif filter_type == "amplitude_filter":
            result = fftconvolve(np.atleast_2d(padded_data), np.atleast_2d(self.a_filter), **kwargs)
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(filter_type))
        return result

    def characterize_traces(self, smoothing=False):
        """
        Create metrics that can be used to mask the trace data.
        Args:
            smoothing: boolean
                Use smoothing splines to calculate the minimum of the
                derivative after the peak. The pulse.noise must exist
                and the phase and amplitude traces must have been
                computed.
        """
        n_samples = self.p_trace.shape[1]
        peak_offset = 10
        data = np.array([self.p_trace, self.a_trace])
        data -= np.mean(data, axis=-1, keepdims=True)
        # determine the mean of the trace prior to the pulse
        self._prepulse_mean = np.zeros(self.peak_indices.shape)
        for index, peak in enumerate(self.peak_indices):
            if peak < 2 * peak_offset:
                self._prepulse_mean[index] = np.inf
            else:
                prepulse = data[:, index, :peak - peak_offset].sum(axis=0)
                self._prepulse_mean[index] = np.mean(prepulse)

        # determine the rms value of the trace prior to the pulse
        self._prepulse_rms = np.zeros(self.peak_indices.shape)
        for index, peak in enumerate(self.peak_indices):
            if peak < 2 * peak_offset:
                self._prepulse_rms[index] = np.inf
            else:
                prepulse = data[:, index, :peak - peak_offset].sum(axis=0)
                self._prepulse_rms[index] = np.sqrt(np.mean(prepulse ** 2))

        # determine the minimum slope after the pulse peak
        self._postpulse_min_slope = np.zeros(self.peak_indices.shape)
        if smoothing:
            sigma = np.std(np.array([self.noise.p_trace, self.noise.a_trace]).sum(axis=0), ddof=1)
            for index, peak in enumerate(self.peak_indices):
                if peak + 2 * peak_offset > n_samples - 1:
                    self._postpulse_min_slope[index] = -np.inf
                else:
                    postpulse = data[:, index, peak + peak_offset:].sum(axis=0)
                    weights = np.empty(postpulse.size)
                    weights.fill(1 / sigma)
                    time = np.linspace(0, postpulse.size / self.sample_rate, postpulse.size) * 1e6
                    s = UnivariateSpline(time, postpulse, w=weights)
                    self._postpulse_min_slope[index] = np.min(s.derivative()(time))
        else:
            for index, peak in enumerate(self.peak_indices):
                if peak + 2 * peak_offset > n_samples - 1:
                    self._postpulse_min_slope[index] = -np.inf
                else:
                    postpulse = data[:, index, peak + peak_offset:].sum(axis=0)
                    self._postpulse_min_slope[index] = np.min(np.diff(postpulse)) * self.sample_rate * 1e6

    def mask_peak_indices(self, minimum, maximum):
        """
        Add traces with peak indices outside of the minimum and maximum to the
        pulse.mask.
        Args:
            minimum: float
                The minimum acceptable peak index
            maximum: float
                The maximum acceptable peak index
        """
        logic = np.logical_or(self.peak_indices < minimum, self.peak_indices > maximum)
        self.mask[logic] = False

    def mask_prepulse_mean(self, minimum, maximum):
        """
        Add traces with pre-pulse means outside of the minimum and maximum to the
        pulse.mask.
        Args:
            minimum: float
                The minimum acceptable pre-pulse mean
            maximum: float
                The maximum acceptable pre-pulse mean
        """
        if self._prepulse_mean is None:
            raise AttributeError("The pulse traces have not been characterized yet.")
        logic = np.logical_or(self._prepulse_mean < minimum, self._prepulse_mean > maximum)
        self.mask[logic] = False

    def mask_prepulse_rms(self, maximum):
        """
        Add traces with pre-pulse RMSs larger than the maximum to the pulse.mask.
        Args:
            maximum: float
                The maximum acceptable pre-pulse rms
        """
        if self._prepulse_rms is None:
            raise AttributeError("The pulse traces have not been characterized yet.")
        logic = self._prepulse_rms > maximum
        self.mask[logic] = False

    def mask_postpulse_min_slope(self, minimum):
        """
        Add traces with post-pulse minimum slopes smaller than the minimum to the
        pulse.mask.
        Args:
            minimum: float
                The minimum acceptable post-pulse minimum slope
        """
        if self._postpulse_min_slope is None:
            raise AttributeError("The pulse traces have not been characterized yet.")
        logic = self._postpulse_min_slope < minimum
        self.mask[logic] = False

    def compute_spectrum(self, use_mask=True, use_calibration=True, calibrated=None, **kwargs):
        """
        Compute the spectrum of the pulse data. The result is stored in
        pulse.spectrum.
        Args:
            use_mask: boolean (optional)
                Use the pulse.mask to determine which detector responses to
                use. The default is True.
            use_calibration: boolean (optional)
                Use the response calibration in pulse.loop to calculate the
                energies. If False, the responses are used directly. The
                default is True.
            calibrated: boolean (optional)
                Determines whether the spectrum is saved as 'calibrated' for
                plotting and resolving_power purposes. This has no effect on
                the use_calibration parameter. The default is None which
                corresponds to the state of use_calibration. Setting this
                parameter is useful for when the responses are already
                energies.
            kwargs: optional keyword arguments
                Optional arguments to scipy.stats.gaussian_kde
        """
        self.clear_spectrum()
        # compute an estimate of the distribution function for the amplitude data
        if use_calibration:
            calibration = self.loop.energy_calibration
            energies = calibration(self.responses[self.mask]) if use_mask else calibration(self.responses)
        else:
            energies = self.responses[self.mask] if use_mask else self.responses

        calibrated = use_calibration if calibrated is None else calibrated
        fwhm, peak, pdf, pdf_interp = self._compute_fwhm(energies, **kwargs)
        self._spectrum = {"pdf": pdf, "interpolation": pdf_interp, "energies": energies, "calibrated": calibrated,
                          "bandwidth": pdf.factor, "fwhm": fwhm, "peak": peak}

    @staticmethod
    def _compute_fwhm(energies, **kwargs):
        pdf = gaussian_kde(energies, **kwargs)
        maximum, minimum = energies.max(), energies.min()
        x = np.linspace(minimum, maximum, int(10 * (maximum - minimum) / pdf.factor))  # sample at 10x the bandwidth
        # convert to a spline so that we can robustly compute the FWHM and maximum later
        # noinspection PyArgumentList
        pdf_interp = InterpolatedUnivariateSpline(x, pdf(x), k=3)
        # compute the maximum of the distribution
        pdf_max = 0
        peak_location = 0
        for root in quadratic_spline_roots(pdf_interp.derivative()):
            if pdf_interp(root) > pdf_max:
                pdf_max = pdf_interp(root)
                peak_location = root.item()
        peak = peak_location if pdf_max != 0 and peak_location != 0 else np.nan
        # compute the FWHM
        # noinspection PyArgumentList
        pdf_approx_shifted = InterpolatedUnivariateSpline(x, pdf(x) - pdf_max / 2, k=3)

        roots = pdf_approx_shifted.roots()
        if roots.size >= 2 and pdf_max != 0 and peak_location != 0:
            # assert roots.size >= 2, "The distribution doesn't have a FWHM."
            indices = np.argsort(np.abs(roots - peak_location))
            roots = roots[indices[:2]]
            fwhm = roots.max() - roots.min()
        else:
            fwhm = np.nan
        return fwhm, peak, pdf, pdf_interp

    def _set_directory(self, directory):
        self._directory = directory
        try:
            self.loop._directory = self._directory
        except AttributeError:
            pass
        try:
            self.noise._directory = self._directory
        except AttributeError:
            pass

    def _threshold_cut(self, use_filter=False, threshold=5):
        """
        Remove traces from the data object that don't meet the threshold condition on the
        phase trace.

        threshold is the number of standard deviations to put the threshold cut.
        """
        if use_filter:
            data = np.array([-self._p_trace_filtered, self._traces[1]])
        else:
            data = self._traces
        # Compute the median average deviation use that to calculate the standard
        # deviation. This should be robust against the outliers from pulses.
        median_phase = np.median(data[0], axis=1, keepdims=True)
        mad = np.median(np.abs(data[0] - median_phase))
        sigma = 1.4826 * mad

        # look for a phase < sigma * threshold around the middle of the trace
        n_points = len(data[0, 0, :])
        middle = (n_points + 1) / 2
        start = int(middle - 25)
        stop = int(middle + 25)
        phase_middle = data[0, :, start:stop]
        indices = np.where((phase_middle - median_phase < -sigma * threshold).any(axis=1))
        self._traces = self._traces[:, indices[0], :]
        if self._traces.size == 0:
            raise RuntimeError('All data was removed by cuts')
        if use_filter:
            self._p_trace_filtered = self._p_trace_filtered[indices[0], :]

    def _offset_correction(self, fractional_offset=True):
        """
        Correct for trigger offsets.
        """
        # pull out filtered data
        data = np.array([-self._p_trace_filtered, self._traces[1]])

        # define frequency vector
        f = fft.rfftfreq(len(data[0, 0, :]))

        # find the middle index based on the minimum phase
        middle_ind = np.median(np.argmin(data[0, :, :], axis=1))

        # loop through triggers and shift the peaks
        for index in range(len(data[0, :, 0])):
            # pull out phase and amplitude
            phase_trace = self._traces[0, index, :]
            amp_trace = self._traces[1, index, :]
            # find the peak index of filtered phase data
            peak_ind = np.argmin(data[0, index, 2:-2]) + 2
            # determine the shift
            if fractional_offset:
                peak = data[0, index, peak_ind - 2: peak_ind + 3]
                poly = np.polyfit([-2, -1, 0, 1, 2], peak, 2)
                frac = -poly[1] / (2 * poly[0])
                shift = middle_ind - peak_ind - frac
            else:
                shift = middle_ind - peak_ind
            # remove the shift by applying a phase
            # same as np.roll(data, shift) for integer offsets
            self._traces[0, index, :] = fft.irfft(fft.rfft(phase_trace) * np.exp(-1j * 2 * np.pi * shift * f),
                                                  len(phase_trace))
            self._traces[1, index, :] = fft.irfft(fft.rfft(amp_trace) * np.exp(-1j * 2 * np.pi * shift * f),
                                                  len(amp_trace))

    def _average_pulses(self):
        """
        Average the data together by summing and normalizing the phase pulse height to 1.
        """
        # add all the data together
        self.template = np.sum(self._traces, axis=1)
        # remove any small baseline error from front of pulses
        self.template = self._remove_baseline(self.template)
        # normalize phase pulse height to 1
        self.template /= np.abs(np.min(self.template[0, :]))

    @staticmethod
    def _remove_baseline(traces):
        """
        Remove the baseline from traces using the first 20% of the data.
        Note: only use when averaging data together since this adds a lot of
        variance to the pulse height.
        """
        ind = int(np.floor(traces.shape[-1] / 5))
        traces -= np.median(traces[..., :ind], axis=-1, keepdims=True)
        return traces

    def plot_template(self):
        """
        Plot the template.
        """
        time = np.linspace(0, self.i_trace.shape[1] / self.sample_rate, self.i_trace.shape[1]) * 1e6
        figure = plt.figure(figsize=(12, 4))
        ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax0.plot(time, self.template[0], 'b-', linewidth=2, label='template data')
        ax0.set_ylabel('phase')
        ax0.set_xlabel(r'time [$\mu$s]')

        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        ax2.plot(time, self.template[1], 'b-', linewidth=2, label='template data')
        ax2.set_ylabel('amplitude')
        ax2.set_xlabel(r'time [$\mu$s]')
        figure.tight_layout()

    def plot_traces(self, calibrate=False, use_mask=True, label="best", fit_type="lmfit", axes_list=None):
        """
        Plot the trace data.
        Args:
            calibrate: boolean
                Boolean that determines if calibrated data is used or not for
                the plot. The default is False.
            use_mask: boolean (optional)
                Use the pulse.mask to determine which traces to plot. The
                default is True.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes on which to put the plots. The default
                is None and a new figure is made. The list must be of length 3
                or of length 1 if only the complex plane is to be plotted.
        Returns:
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes with the plotted data.
            indexer: custom class
                A class that controls the interactive features of the plot.
                Note: this variable must stay in the current namespace for the
                plot to remain interactive.
        """
        # get the time, loop, and traces in complex form for potential calibration
        time = np.linspace(0, self.i_trace.shape[1] / self.sample_rate, self.i_trace.shape[1]) * 1e6  # in µs
        z = self.loop.z
        f = self.loop.f
        mask = self.mask if use_mask else np.ones(self.mask.size, dtype=bool)
        traces = self.i_trace[mask, :] + 1j * self.q_trace[mask, :]
        # grab the model
        _, result_dict = self.loop._get_model(fit_type, label)
        if result_dict is not None:
            params = result_dict['result'].params
            model = result_dict['model']
            f_fit = np.linspace(np.min(f), np.max(f), np.size(f) * 10)
            z_fit = model.model(params, f_fit)
            if calibrate:
                f_traces = np.empty(traces.shape)
                f_traces.fill(self.f_bias)
                traces = model.calibrate(params, traces, f_traces)
                z_fit = model.calibrate(params, z_fit, f_fit)
                z = model.calibrate(params, z, f)
        else:
            z_fit = np.array([])
        # set up figure
        if axes_list is None:
            if result_dict is not None:
                figure = plt.figure(figsize=(6, 8))
                axes_list = [plt.subplot2grid((4, 1), (2, 0), rowspan=2),
                             plt.subplot2grid((4, 1), (0, 0)), plt.subplot2grid((4, 1), (1, 0))]
            else:
                figure, axes = plt.subplots()
                axes_list = [axes]

        else:
            assert len(axes_list) == 3, "axes_list must have length 3"
            figure = axes_list[0].figure

        axes_list[0].plot(z.real, z.imag, 'o', markersize=2)
        loop, = axes_list[0].plot(traces[0, :].real, traces[0, :].imag, 'o', markersize=2)
        axes_list[0].plot(z_fit.real, z_fit.imag)
        axes_list[0].axis('equal')
        axes_list[0].set_xlabel('I [Volts]')
        axes_list[0].set_ylabel('Q [Volts]')

        if result_dict is not None and len(axes_list) > 1:
            try:
                p_trace = self.p_trace[mask, :] * 180 / np.pi
                a_trace = self.a_trace[mask, :]
                axes_list[1].set_ylabel("phase [degrees]")
                axes_list[2].set_ylabel("amplitude [radians]")
            except AttributeError:
                p_trace = self.i_trace[mask, :]
                a_trace = self.q_trace[mask, :]
                axes_list[1].set_ylabel("I [V]")
                axes_list[2].set_ylabel("Q [V]")
            phase, = axes_list[1].plot(time, p_trace[0, :])
            amp, = axes_list[2].plot(time, a_trace[0, :])
            axes_list[2].set_xlabel(r"time [$\mu s$]")

        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15)

        class Index(object):
            def __init__(self, ax_slider, ax_prev, ax_next):
                self.ind = 0
                self.num = len(traces[:, 0])
                self.bnext = Button(ax_next, 'Next')
                self.bnext.on_clicked(self.next)
                self.bprev = Button(ax_prev, 'Previous')
                self.bprev.on_clicked(self.prev)
                self.slider = Slider(ax_slider, 'Trace Index: ', 0, self.num, valinit=0, valfmt='%d')
                self.slider.on_changed(self.update)

                self.slider.label.set_position((0.5, -0.5))
                self.slider.valtext.set_position((0.5, -0.5))

            def next(self, event):
                log.debug(event)
                self.ind += 1
                i = self.ind % self.num
                self.slider.set_val(i)

            def prev(self, event):
                log.debug(event)
                self.ind -= 1
                i = self.ind % self.num
                self.slider.set_val(i)

            def update(self, value):
                self.ind = int(value)
                i = self.ind % self.num
                loop.set_xdata(traces[i, :].real)
                loop.set_ydata(traces[i, :].imag)
                phase.set_ydata(p_trace[i, :])
                amp.set_ydata(a_trace[i, :])

                axes_list[1].relim()
                axes_list[1].autoscale()
                axes_list[2].relim()
                axes_list[2].autoscale()
                plt.draw()

        position = axes_list[2].get_position()
        slider = plt.axes([position.x0, 0.05, position.width / 2, 0.03])
        middle = position.x0 + 3 * position.width / 4
        prev = plt.axes([middle - 0.18, 0.05, 0.15, 0.03])
        next_ = plt.axes([middle + 0.02, 0.05, 0.15, 0.03])
        indexer = Index(slider, prev, next_)

        return axes_list, indexer

    def plot_metrics(self, use_mask=True):
        """
        Plot the metrics created by pulse.characterize_traces() in a corner plot.
        Args:
            use_mask: boolean (optional)
                Use the pulse.mask to determine which traces to plot metrics for.
                The default is True.
        """
        if self._prepulse_mean is None or self._prepulse_rms is None or self._postpulse_min_slope is None:
            raise AttributeError("Data metrics have not been computed yet.")
        if use_mask:
            metrics = np.vstack([self.peak_indices[self.mask], self._prepulse_mean[self.mask],
                                 self._prepulse_rms[self.mask], self._postpulse_min_slope[self.mask]]).T
        else:
            metrics = np.vstack([self.peak_indices, self._prepulse_mean,
                                 self._prepulse_rms, self._postpulse_min_slope]).T
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)  # for singular axis limits
            corner.corner(metrics, labels=['peak times', 'prepulse mean', 'prepulse rms', 'postpulse min slope'],
                          plot_contours=False, plot_density=False, range=[.97, .97, .97, .97], bins=[100, 20, 20, 100],
                          quiet=True)

    def plot_spectrum(self, x_limits=None, second_x_axis=False, axes=None):
        """
        Plot the spectrum of the pulse responses.
        Args:
            x_limits: length 2 iterable of floats
                Bounds on the x-axis
            second_x_axis: boolean
                If True, a second x-axis is plotted below the first with the
                wavelength values. The default is False.
            axes: matplotlib.axes.Axes class
                An axes class for plotting the data.
        Returns:
            axes: matplotlib.axes.Axes class
                An axes class with the plotted data.
        """
        # get the needed data from the spectrum dictionary
        pdf = self.spectrum["pdf"]
        energies = self.spectrum["energies"]
        bandwidth = self.spectrum["bandwidth"]
        calibrated = self.spectrum["calibrated"]
        # use the known energy if possible
        peak = self.energies[0] if len(self.energies) == 1 and calibrated else self.spectrum["peak"]
        fwhm = self.spectrum["fwhm"]
        min_energy = energies.min()
        max_energy = energies.max()

        # get the figure axes
        if not axes:
            figure, axes = plt.subplots()
        else:
            figure = axes.figure

        # plot the data
        n_bins = 10 * int((max_energy - min_energy) / bandwidth)
        axes.hist(energies, n_bins, density=True)
        xx = np.linspace(min_energy, max_energy, 10 * n_bins)
        label = "R = {:.2f}".format(peak / fwhm) if not np.isnan(peak) and not np.isnan(fwhm) else ""
        axes.plot(xx, pdf(xx), 'k-', label=label)

        # set x axis limits
        if x_limits is not None:
            axes.set_xlim(x_limits)
        else:
            axes.set_xlim([min_energy, max_energy])

        # format figure
        axes.set_xlabel('energy [eV]') if calibrated else axes.set_xlabel("response [radians]")
        axes.set_ylabel('probability density')
        if label:
            axes.legend()

        # put twin axis on the bottom
        if second_x_axis:
            wvl_axes = axes.twiny()
            wvl_axes.set_frame_on(True)
            wvl_axes.patch.set_visible(False)
            wvl_axes.xaxis.set_ticks_position('bottom')
            wvl_axes.xaxis.set_label_position('bottom')
            wvl_axes.spines['bottom'].set_position(('outward', 40))
            wvl_axes.set_xlabel('wavelength [nm]')
            if x_limits is not None:
                wvl_axes.set_xlim(x_limits)
            else:
                wvl_axes.set_xlim([min_energy, max_energy])

            # redo ticks on bottom axis
            def tick_labels(x):
                v = ev_nm_convert(x)
                return ["%.0f" % z for z in v]
            x_locs = axes.xaxis.get_majorticklocs()
            wvl_axes.set_xticks(x_locs)
            wvl_axes.set_xticklabels(tick_labels(x_locs))

        figure.tight_layout()
        return axes
