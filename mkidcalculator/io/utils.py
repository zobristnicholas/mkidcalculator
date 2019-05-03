import os
import logging
import tempfile
import numpy as np
import scipy.constants as c

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class NpzHolder:
    """Loads npz file when requested and saves them."""
    def __init__(self):
        self._files = {}

    def __getitem__(self, item):
        # if string load and save to cache
        if isinstance(item, str):
            item = os.path.abspath(item)
            # check if already loaded
            if item in self._files.keys():
                log.info("loaded from cache: {}".format(item))
                return self._files[item]
            else:
                npz = np.load(item)
                log.info("loaded: {}".format(item))
                self._files[item] = npz
                log.info("saved to cache: {}".format(item))
                return self._files[item]
        # if NpzFile skip loading but save if it hasn't been loaded before
        elif isinstance(item, np.lib.npyio.NpzFile):
            file_name = os.path.abspath(item.fid.name)
            if file_name not in _loaded_npz_files.keys():
                log.info("loaded: {}".format(file_name))
                self._files[file_name] = item
                log.info("saved to cache: {}".format(file_name))
            else:
                log.info("loaded from cache: {}".format(file_name))
            return item
        elif item is None:
            return None
        else:
            raise ValueError("'item' must be a valid file name or a numpy npz file object.")

    def free_memory(self, file_names=None):
        """
        Removes file names in file_names from active memory. If file_names is
        None, all are removed (default).
        """
        if file_names is None:
            file_names = self._files.keys()
        elif isinstance(file_names, str):
            file_names = [file_names]
        for file_name in file_names:
            npz = self._files.pop(file_name, None)
            del npz


_loaded_npz_files = NpzHolder()  # cache of already loaded files


def compute_phase_and_amplitude(cls, label="best", fit_type="lmfit", fr=None, center=None, unwrap=True):
    """
    Compute the phase and amplitude traces stored in pulse.p_trace and
    pulse.a_trace.
    Args:
        cls: Pulse or Noise class
            The Pulse or Noise class used to create the phase and amplitude
            data.
        label: string
            Corresponds to the label in the loop.lmfit_results or
            loop.emcee_results dictionaries where the fit parameters are.
            The resulting DataFrame is stored in
            object.loop_parameters[label]. The default is "best", which gets
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
    # clear prior data
    cls.clear_traces()
    # get the model and parameters
    _, result_dict = cls.loop._get_model(fit_type, label)
    model = result_dict["model"]
    params = result_dict["result"].params
    # get the resonance frequency and loop center
    fr = params["fr"].value if fr is None else params[fr].value
    if center is None:
        center = "1 - q0 / (2 * qc) - 1j * q0**2 / qc * df / f0"
    center = params._asteval.eval(center)
    # get complex IQ data for the traces and loop at the resonance frequency
    traces = cls.i_trace + 1j * cls.q_trace
    z_fr = model.model(params, fr)
    f = np.empty(traces.shape)
    f.fill(cls.f_bias)
    # calibrate the IQ data
    traces = model.calibrate(params, traces, f)
    z_fr = model.calibrate(params, z_fr, fr)
    # center and rotate the IQ data
    traces = (center - traces)
    z_fr = (center - z_fr)  # should be real if no loop asymmetry
    # compute the phase and amplitude traces from the centered traces
    cls.p_trace = np.unwrap(np.angle(traces) - np.angle(z_fr)) if unwrap else np.angle(traces) - np.angle(z_fr)
    cls.a_trace = np.abs(traces) / np.abs(z_fr) - 1


def offload_data(cls, excluded_keys=(), npz_key="_npz", prefix="", directory_key="_directory"):
    """
    Offload data in excluded_keys from the class to an npz file. The npz file
    name is stored in cls.npz_key.
    Args:
        cls: class
            The class being unpickled
        excluded_keys: iterable of strings
            Keys to force into npz format. The underlying attributes must be
            numpy arrays. The default is to not exclude any keys.
        npz_key: string
            The class attribute name that corresponds to where the npz file was
            stored. The default is "_npz".
        prefix: string
            File name prefix for the class npz file if a new one needs to be
            made. The default is no prefix.
        directory_key: string
            The class attribute that corresponds to the data directory. If it
            doesn't exist, than the current directory is used.
    Returns:
        cls.__dict__: dictionary
            The new class dict which can be used for pickling.
    """
    # get the directory
    directory = "." if getattr(cls, directory_key, None) is None else getattr(cls, directory_key)
    directory = os.path.abspath(directory)
    # if we've overloaded any excluded key, aren't using the npz file yet, or are changing directories (re)make npz
    make_npz = False
    for key in excluded_keys:
        make_npz = make_npz or isinstance(getattr(cls, key), np.ndarray)
    if isinstance(getattr(cls, npz_key), str):
        file_name = getattr(cls, npz_key)
        if os.path.dirname(file_name) != directory:
            make_npz = True
    if make_npz:
        # get the data to save
        excluded_data = {}
        for key in excluded_keys:
            if getattr(cls, key) is not None:
                excluded_data[key] = getattr(cls, key)
        # if there is data to save, save it
        if excluded_data:
            # get the npz file name
            file_name = tempfile.mkstemp(prefix=prefix, suffix=".npz", dir=directory)[1]
            np.savez(file_name, **excluded_data)
            setattr(cls, npz_key, file_name)
    # change the excluded keys in the dict to the key for the npz_file if it exists
    if getattr(cls, npz_key) is not None:
        for key in excluded_keys:
            cls.__dict__[key] = key
    return cls.__dict__


def quadratic_spline_roots(spline):
    """Returns the roots of a scipy spline."""
    roots = []
    knots = spline.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spline(a), spline((a + b) / 2), spline(b)
        t = np.roots([u + w - 2 * v, w - u, 2 * v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t * (b - a) / 2 + (b + a) / 2)
    return np.array(roots)


def ev_nm_convert(x):
    """
    If x is a wavelength in nm, the corresponding energy in eV is returned.
    If x is an energy in eV, the corresponding wavelength in nm is returned.
    """
    return c.speed_of_light * c.h / c.eV * 1e9 / x
