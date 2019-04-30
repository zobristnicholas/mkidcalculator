import pickle
import logging
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
from scipy import signal as sg
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider

from mkidcalculator.io.data import AnalogReadoutPulse
from mkidcalculator.io.utils import compute_phase_and_amplitude

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
        self._p_filter = None
        self._a_filter = None
        # data products
        self._amplitudes = None
        log.info("Pulse object created. ID: {}".format(id(self)))

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
        self._noise = noise
        self.clear_noise_data()
        try:
            self.noise.loop = self.loop
        except AttributeError:
            pass

    @property
    def amplitudes(self):
        """
        A settable property that contains the detector amplitudes made with
        pulse.compute_amplitudes().
        """
        if self._amplitudes is None:
            raise AttributeError("The amplitudes for this pulse have not been calculated yet.")
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, amplitudes):
        self._amplitudes = amplitudes

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
        self._template = template

    @property
    def optimal_filter(self):
        """
        A settable property that contains the optimal filter made with
        pulse.make_filters().
        """
        if self._optimal_filter is None:
            raise AttributeError("The optimal filter for this pulse has not been calculated yet.")
        return self._optimal_filter

    @optimal_filter.setter
    def optimal_filter(self, optimal_filter):
        self._optimal_filter = optimal_filter

    @property
    def p_filter(self):
        """
        A settable property that contains the phase filter made with
        pulse.make_filters().
        """
        if self._p_filter is None:
            raise AttributeError("The phase filter for this pulse has not been calculated yet.")
        return self._p_filter

    @p_filter.setter
    def p_filter(self, p_filter):
        self._p_filter = p_filter

    @property
    def a_filter(self):
        """
        A settable property that contains the amplitude filter made with
        pulse.make_filters().
        """
        if self._a_filter is None:
            raise AttributeError("The amplitude filter for this pulse has not been calculated yet.")
        return self._a_filter

    @a_filter.setter
    def a_filter(self, a_filter):
        self._a_filter = a_filter

    def clear_loop_data(self):
        """Remove all data calculated from the pulse.loop attribute."""
        self.a_trace = None
        self.p_trace = None
        self.clear_template()
        self.clear_filters()

    def clear_noise_data(self):
        """Remove all data calculated from the pulse.noise attribute."""
        self.clear_template()
        self.clear_filters()

    def clear_template(self):
        """
        Clear the template made with pulse.make_template() and reset the
        traces used to make the template.
        """
        try:
            self._traces = self._remove_baseline(np.array([self.p_trace, self.a_trace]))
        except AttributeError:
            pass
        self.template = None

    def clear_filters(self):
        """Clear the filters made with pulse.make_filters()."""
        self._p_trace_filtered = None
        self.optimal_filter = None
        self.p_filter = None
        self.a_filter = None

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", fr=None, center=None, unwrap=True):
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
        """
        compute_phase_and_amplitude(self, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)
        try:
            compute_phase_and_amplitude(self.noise, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)
        except AttributeError:
            pass

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
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

    def make_template(self):
        """
        Make a template from phase and amplitude data. The template is needed
        for computing a filter.
        """
        # create a rough template by cutting the noise traces and averaging
        self.clear_template()
        self._threshold_cut()
        self._average_pulses()
        # make a filter with the template
        self.make_filters()
        # do a better job using a filter
        self.clear_template()
        self._p_trace_filtered = self.apply_filter(self._traces[0], filter_type="phase_filter")
        self._threshold_cut(use_filter=True)
        self._offset_correction()
        self._average_pulses()
        self.clear_filters()  # this filter is not using the updated template so get rid of it

    def make_filters(self):
        """
        Make an optimal filter assuming a linear response and stationary noise.
        A full 2D filter is made as well as two 1D filters for the phase and
        amplitude responses.
        """
        # pull out shape parameters
        n_samples = len(self.template[0])
        shape = (2, n_samples // 2 + 1)
        if self.noise.pp_psd.size != shape[1]:
            raise ValueError("The noise data PSDs must have a shape compatible with the pulse data")
        # compute template fft
        template_fft = fft.rfft(self.template)

        # assemble noise matrix
        s = np.array([[self.noise.pp_psd, self.noise.pa_psd],
                      [np.conj(self.noise.pa_psd), self.noise.aa_psd]], dtype=np.complex)

        # compute the optimal filter: conj(template_fft) @ s_inv
        filter_fft = np.zeros(shape, dtype=np.complex)
        for index in range(shape[1]):
            filter_fft[:, index] = la.lstsq(s[:, :, index].T, np.conj(template_fft[:, index]), rcond=None)[0]
        # return to time domain
        self.optimal_filter = fft.irfft(filter_fft, n_samples)
        # normalize the optimal filter
        norm = (sg.convolve(self.optimal_filter[0], self.template[0], mode='same') +
                sg.convolve(self.optimal_filter[1], self.template[1], mode='same')).max()
        norm /= np.abs(self.template[0].min() + self.template[1].min())  # the templates don't add to one
        self.optimal_filter /= norm

        # compute the phase only optimal filter: conj(phase_fft) / J
        phase_filter_fft = (np.conj(template_fft[0, :]) / self.noise.pp_psd)
        self.p_filter = fft.irfft(phase_filter_fft, n_samples)
        # normalize
        norm = sg.convolve(self.p_filter, self.template[0], mode='same').max()
        norm /= np.abs(self.template[0].min())
        self.p_filter /= norm

        # compute the amplitude only optimal filter: conj(amplitude_fft) / J
        amplitude_filter_fft = (np.conj(template_fft[1, :]) / self.noise.aa_psd)
        self.a_filter = fft.irfft(amplitude_filter_fft, n_samples)
        norm = sg.convolve(self.a_filter, self.template[1], mode='same').max()
        norm /= np.abs(self.template[1].min())
        self.a_filter /= norm

    def variance(self, calculation_type="optimal_filter"):
        """
        Compute the expected variance for a particular response calculation
        type.
        """
        # check for template and grab some constants
        n_samples = self.template.shape[1]
        sample_rate = pulse.sample_rate
        # calculate the variance for requested calculation type
        if calculation_type == "optimal_filter":
            filter_fft = fft.rfft(self.optimal_filter).transpose()  # (N, 2)
            template_fft = fft.rfft(self.template)  # (2, N)
            variance = (sample_rate * n_samples / (4 * np.sum(filter_fft @ template_fft).real)).squeeze()
        elif calculation_type == "phase_filter":
            filter_fft = fft.rfft(self.p_filter)  # (N,)
            template_fft = fft.rfft(self.template[0])  # (N,)
            variance = (sample_rate * n_samples / (4 * np.sum(filter_fft @ template_fft).real)).squeeze()
        elif calculation_type == "amplitude_filter":
            filter_fft = fft.rfft(self.a_filter)  # (N,)
            template_fft = fft.rfft(self.template[1])  # (N,)
            variance = (sample_rate * n_samples / (4 * np.sum(filter_fft @ template_fft).real)).squeeze()
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(calculation_type))

        return variance

    def compute_amplitudes(self, calculation_type="optimal_filter"):
        """
        Compute the detector response amplitudes for a given filter type. The
        results are stored in self.amplitudes.
        """
        if calculation_type == "optimal_filter":
            data = self._remove_baseline(np.array([self.p_trace, self.a_trace]))
            amplitudes = np.empty(data.shape[1])
            for index in range(data.shape[1]):
                amplitudes[index] = (sg.convolve(self.optimal_filter[0], data[0, index, :], mode='same') +
                                     sg.convolve(self.optimal_filter[1], data[1, index, :], mode='same')).max()
        elif calculation_type == "phase_filter":
            data = self._remove_baseline(self.p_trace)
            amplitudes = np.empty(data.shape[0])
            for index in range(data.shape[0]):
                amplitudes[index] = sg.convolve(self.p_filter, data[index, :], mode='same').max()
        elif calculation_type == "amplitude_filter":
            data = self._remove_baseline(self.a_trace)
            amplitudes = np.empty(data.shape[0])
            for index in range(data.shape[0]):
                amplitudes[index] = sg.convolve(self.a_filter[0], data[index, :], mode='same').max()
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(calculation_type))
        self.amplitudes = amplitudes

    def apply_filter(self, data, filter_type="optimal_filter"):
        """
        Method for convolving the two dimensional filter with the data. The data can
        either be a 2xN matrix or a 2xMxN matrix where N is the trace length and M is the
        number of traces.
        """
        if filter_type == "optimal_filter":
            if data.shape == self.optimal_filter.shape:
                result = (sg.convolve(self.optimal_filter[0], data[0], mode='same') +
                          sg.convolve(self.optimal_filter[1], data[1], mode='same'))
            elif len(data.shape) == 3 and data.shape[0] == 2 and data.shape[2] == len(self.optimal_filter[0]):
                result = np.empty(data.shape[1:])
                for index in range(data.shape[1]):
                    result[index, :] = (sg.convolve(self.optimal_filter[0], data[0, index, :], mode='same') +
                                        sg.convolve(self.optimal_filter[1], data[1, index, :], mode='same'))
            else:
                raise ValueError("data needs to be a 2 x N x M array (last dimension optional)")
        elif filter_type == "phase_filter":
            if data.shape == self.p_filter.shape:
                result = sg.convolve(self.p_filter, data, mode='same')
            elif len(data.shape) == 2 and data.shape[1] == len(self.p_filter):
                result = np.empty(data.shape)
                for index in range(data.shape[0]):
                    result[index, :] = sg.convolve(self.p_filter, data[index, :], mode='same')
            else:
                raise ValueError("data needs to be a 1 or 2D array with the last dimension "
                                 "equal in length to the filter length")
        elif filter_type == "amplitude_filter":
            if data.shape == self.a_filter.shape:
                result = sg.convolve(self.a_filter, data, mode='same')
            elif len(data.shape) == 2 and data.shape[1] == len(self.a_filter):
                result = np.empty(data.shape)
                for index in range(data.shape[0]):
                    result[index, :] = sg.convolve(self.a_filter, data[index, :], mode='same')
            else:
                raise ValueError("data needs to be a 1 or 2D array with the last dimension "
                                 "equal in length to the filter length")
        else:
            raise ValueError("'{}' is not a valid calculation_type".format(filter_type))
        return result

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
        """
        ind = int(np.floor(traces.shape[-1] / 5))
        traces -= np.median(traces[..., :ind], axis=-1, keepdims=True)
        return traces

    def plot_template(self):
        """
        Plot the template
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

    def plot_traces(self, calibrate=False, label="best", fit_type="lmfit", axes_list=None):
        """
        Plot the trace data.
        Args:
            calibrate: boolean
                Boolean that determines if calibrated data is used or not for
                the plot. The default is False.
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
        traces = self.i_trace + 1j * self.q_trace
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
                p_trace = self.p_trace * 180 / np.pi
                a_trace = self.a_trace
                axes_list[1].set_ylabel("phase [degrees]")
                axes_list[2].set_ylabel("amplitude [radians]")
            except AttributeError:
                p_trace = self.i_trace
                a_trace = self.q_trace
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
