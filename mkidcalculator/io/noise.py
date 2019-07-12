import os
import pickle
import logging
import numpy as np
from scipy.signal import welch, csd
from matplotlib import pyplot as plt

from mkidcalculator.io.data import AnalogReadoutNoise
from mkidcalculator.io.utils import compute_phase_and_amplitude, offload_data, _loaded_npz_files

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    """A class for manipulating the noise data."""
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        # loop reference for computing phase and amplitude
        self._loop = None
        # phase and amplitude data
        self._p_trace = None
        self._a_trace = None
        # noise data
        self._f_psd = None
        self._ii_psd = None
        self._qq_psd = None
        self._iq_psd = None
        self._pp_psd = None
        self._aa_psd = None
        self._pa_psd = None
        self._n_samples = None  # for generate_noise()
        # for holding large data
        self._npz = None
        self._directory = None
        log.debug("Noise object created. ID: {}".format(id(self)))

    def __getstate__(self):
        return offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="noise_data_")

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
    def metadata(self):
        """A dictionary containing metadata about the noise."""
        return self._data["metadata"]

    @property
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def sample_rate(self):
        """The sample rate of the IQ data."""
        return self._data['sample_rate']

    @property
    def loop(self):
        """
        A settable property that contains the Loop object required for doing
        noise calculations like computing the phase and amplitude traces. If
        the loop has not been set, it will raise an AttributeError. When the
        loop is set, all information created from the previous loop is deleted.
        """
        if self._loop is None:
            raise AttributeError("The loop object for this noise has not been set yet.")
        return self._loop

    @loop.setter
    def loop(self, loop):
        if self._loop is not loop:
            self.clear_loop_data()
        self._loop = loop

    @property
    def p_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() is run.
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
        noise.compute_phase_and_amplitude() is run.
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
    def f_psd(self):
        """
        A settable property that contains the noise frequency information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._f_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._f_psd

    @f_psd.setter
    def f_psd(self, f_psd):
        self._f_psd = f_psd

    @property
    def ii_psd(self):
        """
        A settable property that contains the II noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._ii_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._ii_psd

    @ii_psd.setter
    def ii_psd(self, ii_psd):
        self._ii_psd = ii_psd

    @property
    def qq_psd(self):
        """
        A settable property that contains the QQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._qq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._qq_psd

    @qq_psd.setter
    def qq_psd(self, qq_psd):
        self._qq_psd = qq_psd

    @property
    def iq_psd(self):
        """
        A settable property that contains the IQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._iq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._iq_psd

    @iq_psd.setter
    def iq_psd(self, iq_psd):
        self._iq_psd = iq_psd

    @property
    def pp_psd(self):
        """
        A settable property that contains the PP noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._pp_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._pp_psd

    @pp_psd.setter
    def pp_psd(self, pp_psd):
        self._pp_psd = pp_psd

    @property
    def aa_psd(self):
        """
        A settable property that contains the AA noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._aa_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._aa_psd

    @aa_psd.setter
    def aa_psd(self, aa_psd):
        self._aa_psd = aa_psd

    @property
    def pa_psd(self):
        """
        A settable property that contains the PA noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._pa_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._pa_psd

    @pa_psd.setter
    def pa_psd(self, pa_psd):
        self._pa_psd = pa_psd

    def clear_loop_data(self):
        """Remove all data calculated from the noise.loop attribute."""
        self.clear_traces()
        self.aa_psd = None
        self.pp_psd = None
        self.pa_psd = None

    def clear_traces(self):
        """
        Remove all trace data calculated from noise.i_trace and noise.q_trace.
        """
        self.a_trace = None
        self.p_trace = None
        self.free_memory()
        self._npz = None

    def free_memory(self, directory=None):
        """
        Offloads a_traces and p_traces to an npz file if they haven't been
        offloaded already and removes any npz file objects from memory, keeping
        just the file name. It doesn't do anything if they don't exist.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the noise was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="noise_data_")
        if isinstance(self._npz, str):  # there might not be an npz file yet
            _loaded_npz_files.free_memory(self._npz)
        try:
            self._data.free_memory()
        except AttributeError:
            pass

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
        log.info("saved noise as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Noise class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            noise = pickle.load(f)
        assert isinstance(noise, cls), "'{}' does not contain a Noise class.".format(file_name)
        log.info("loaded noise from '{}'".format(file_name))
        return noise

    @classmethod
    def load(cls, noise_file_name, data=AnalogReadoutNoise, **kwargs):
        """
        Noise class factory method that returns a Noise() with the data loaded.
        Args:
            noise_file_name: string
                The file name for the noise data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            noise: object
                A Noise() object containing the loaded data.
        """
        noise = cls()
        noise._data = data(noise_file_name, **kwargs)
        return noise

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", fr="fr", unwrap=True):
        """
        Compute the phase and amplitude traces stored in noise.p_trace and
        noise.a_trace.
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
                The default is "fr" which gives the resonance frequency for the
                mkidcalculator.S21 model. This parameter determines the zero
                point for the traces.
            unwrap: boolean
                Determines whether or not to unwrap the phase data. The default
                is True.
        """
        compute_phase_and_amplitude(self, label=label, fit_type=fit_type, fr=fr, unwrap=unwrap)

    def compute_psd(self, **kwargs):
        """
        Compute the noise power spectral density of the noise data in this object.
        Args:
            kwargs: optional keyword arguments
                keywords for the scipy.signal.welch and scipy.signal.csd
                methods. The spectrum scaling and two-sided spectrum of the PSD
                can not be changed since they are assumed in other methods.
        """
        # update keyword arguments
        noise_kwargs = {'nperseg': self.i_trace.shape[1], 'fs': self.sample_rate, 'return_onesided': True,
                        'detrend': 'constant', 'scaling': 'density'}
        noise_kwargs.update(kwargs)
        assert noise_kwargs['scaling'] == 'density', "The PSD scaling is not an allowed keyword."
        assert noise_kwargs['return_onesided'], "A two-sided PSD is not an allowed keyword."
        # compute I/Q noise in V^2 / Hz
        self.f_psd, ii_psd = welch(self.i_trace, **noise_kwargs)
        _, qq_psd = welch(self.q_trace, **noise_kwargs)
        # scipy has different order convention we use equation 5.2 from J. Gao's 2008 thesis.
        # noise_iq = F(I) conj(F(Q))
        _, iq_psd = csd(self.q_trace, self.i_trace, **noise_kwargs)
        # average multiple PSDs together
        self.ii_psd = np.mean(ii_psd, axis=0)
        self.qq_psd = np.mean(qq_psd, axis=0)
        self.iq_psd = np.mean(iq_psd, axis=0)
        # record n_samples for generate_noise()
        self._n_samples = noise_kwargs['nperseg']
        try:
            # compute phase and amplitude noise in rad^2 / Hz
            _, pp_psd = welch(self.p_trace, **noise_kwargs)
            _, aa_psd = welch(self.a_trace, **noise_kwargs)
            _, pa_psd = csd(self.a_trace, self.p_trace, **noise_kwargs)
            # average multiple PSDs together
            self.pp_psd = np.mean(pp_psd, axis=0)
            self.aa_psd = np.mean(aa_psd, axis=0)
            self.pa_psd = np.mean(pa_psd, axis=0)
        except AttributeError:
            pass

    def generate_noise(self, noise_type="pa", n_traces=10000, psd=None):
        """
        Generate fake noise traces from the computed PSDs.
        Args:
            noise_type: string
                The type of noise to generate. Valid options are "pa", "p",
                "a", "iq", "i", "q", which correspond to phase, amplitude, I,
                and Q.
            n_traces: integer
                The number of noise traces to make.
            psd: length 3 iterable of numpy.arrays
                PSD to use to generate the noise in the form
                (PSD_00, PSD_11, PSD_01). Components that aren't used can be
                set to None.
        Returns:
            noise: np.ndarray
                If noise_type == "pa" (or "iq"), a 2 x n_traces x N array of
                noise is made where the first dimension is phase then
                amplitude or (I then Q).
                If noise_type == "p" or "a" or "i" or "q" a n_traces x N array
                of noise is made.
        """
        # check parameters
        noise_types = ["pa", "p", "a", "iq", "i", "q"]
        if noise_type not in noise_types:
            raise ValueError("'noise_type' is not in {}".format(noise_types))
        # get constants
        dt = 1 / self.sample_rate
        if psd is not None:
            psd_00, psd_11, psd_01 = psd
            if hasattr(psd_00, "size"):
                n_frequencies = psd_00.size
            else:
                n_frequencies = psd_11.size
        elif noise_type in ["pa", "p", "a"]:
            psd_00 = self.pp_psd
            psd_01 = self.pa_psd
            psd_11 = self.aa_psd
            n_frequencies = self.f_psd.size
        else:
            psd_00 = self.ii_psd
            psd_01 = self.iq_psd
            psd_11 = self.qq_psd
            n_frequencies = self.f_psd.size

        if noise_type in ["pa", "iq"]:
            # compute square root of covariance
            c = np.array([[psd_00, psd_01],  # 2 x 2 x n_frequencies
                          [np.conj(psd_01), psd_11]])
            c = np.moveaxis(c, 2, 0)  # n_frequencies x 2 x 2
            u, s, vh = np.linalg.svd(c)
            s = np.array([[s[:, 0], np.zeros(s[:, 0].shape)],
                          [np.zeros(s[:, 0].shape), s[:, 1]]])
            s = np.moveaxis(s, -1, 0)
            a = u @ np.sqrt(self._n_samples * s / (2 * dt)) @ vh  # divide by 2 for single sided noise
            # get unit amplitude random phase noise in both quadratures
            phase_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            phase_fft = np.exp(1j * phase_phi)
            amp_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            amp_fft = np.exp(1j * amp_phi)
            # rescale the noise to the covariance
            noise_fft = np.array([[phase_fft],  # 2 x 1 x n_traces x n_frequencies
                                  [amp_fft]])
            noise_fft = np.moveaxis(noise_fft, [0, 1], [-2, -1])  # n_traces x n_frequencies x 2 x 1
            noise_fft = (a @ noise_fft).squeeze()  # n_traces x n_frequencies x 2
            noise_fft = np.moveaxis(noise_fft, -1, 0)  # 2 x n_traces x n_frequencies
        else:
            # compute square root of covariance
            psd = psd_00 if noise_type in ["p", "i"] else psd_11
            a = np.sqrt(self._n_samples * psd / (2 * dt))
            # get unit amplitude random phase noise
            noise_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            noise_fft = np.exp(1j * noise_phi)  # n_traces x n_frequencies
            # rescale the noise to the covariance
            noise_fft = a * noise_fft
        noise = np.fft.irfft(noise_fft, self._n_samples)
        return noise

    def plot_psd(self, noise_type="iq", axes=None):
        if noise_type not in ["iq", "pa"]:
            raise ValueError("Noise type must be one of 'iq' or 'pa'.")
        # get the figure axes
        if not axes:
            figure, axes = plt.subplots()
        else:
            figure = axes.figure
        iq = (noise_type == "iq")
        psd11 = self.ii_psd if iq else 10 * np.log10(self.pp_psd)
        psd22 = self.qq_psd if iq else 10 * np.log10(self.aa_psd)

        axes.step(self.f_psd[1:-1], psd22[1:-1], where='mid', label="Q" if iq else "dissipation", color="C1")
        axes.step(self.f_psd[1:-1], psd11[1:-1], where='mid', label="I" if iq else "phase", color="C0")

        axes.set_xlim(self.f_psd[1:-1].min(), self.f_psd[1:-1].max())
        axes.set_ylabel('PSD [V² / Hz]' if iq else 'PSD [dBc / Hz]')
        axes.set_xlabel('frequency  [Hz]')
        axes.set_xscale('log')
        if iq:
            axes.set_yscale('log')
        axes.legend()
        figure.tight_layout()

    def _set_directory(self, directory):
        self._directory = directory
        try:
            self.loop._directory = self._directory
        except AttributeError:
            pass
