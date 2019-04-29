import pickle
import logging
import numpy as np
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

    def clear_loop_data(self):
        """Remove all data calculated from the pulse.loop attribute."""
        self.a_trace = None
        self.p_trace = None

    def clear_noise_data(self):
        """Remove all data calculated from the pulse.noise attribute."""
        pass

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

    def compute_trace_energies(self):
        raise NotImplementedError

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
