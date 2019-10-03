import os
import glob
import logging
import numpy as np
from scipy.io import loadmat

from mkidcalculator.io.utils import (_loaded_npz_files, offload_data, ev_nm_convert, load_legacy_binary_data,
                                     structured_to_complex)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def analogreadout_temperature(metadata):
    """
    Returns the average temperature across all the temperatures taken during
    the data set.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        temperature: float
            The average temperature during the data set.
    """
    temperatures = [metadata[key]['thermometer']['temperature'][0]
                    for key in metadata.keys() if key not in ('parameters', 'file_name')]
    temperature = np.mean(temperatures)
    return temperature


def analogreadout_sample_rate(metadata):
    """
    Returns the sample rate in Hz from the metadata.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        sample_rate: float
            The sample rate in Hz.
    """
    sample_rate = metadata['parameters']['sample_rate'] * 1e6
    return sample_rate


class AnalogReadoutABC:
    """
    Abstract base class for handling data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    def __init__(self, npz_handle=None, channel=None, index=None):
        self.channel = channel
        self.index = index
        if isinstance(npz_handle, str):
            self._npz = npz_handle
        elif isinstance(npz_handle, np.lib.npyio.NpzFile):
            self._npz = os.path.abspath(npz_handle.fid.name)
        else:
            self._npz = None

    def __getstate__(self):
        return offload_data(self)

    def __getitem__(self, item):
        # get conversion values
        try:
            convert = self.CONVERT[item]
        except KeyError:
            raise KeyError("allowed keys for this data structure are in {}".format(list(self.CONVERT.keys())))
        try:
            # get the result from the npz file
            result = _loaded_npz_files[self._npz][convert[0] if isinstance(convert, tuple) else convert]
            if result.dtype == np.dtype('O'):
                # if it's an object unpack it
                result = result.item()
            else:
                # else get the channel and index
                if self.index is not None:
                    result = result[:, self.index, ...]
                if self.channel is not None:
                    result = result[self.channel]
            # more conversion
            if isinstance(convert, tuple) and len(convert) > 1:
                # try the first conversion
                if not callable(convert[1]):
                    try:
                        if isinstance(convert[1], (tuple, list)):
                            for c in convert[1]:
                                result = result[c]
                        else:
                            result = result[convert[1]]
                    except IndexError:  # didn't work try the second
                        # if that failed run a function (for complex formatted data)
                        result = convert[2](result)
                else:
                    result = convert[1](result)
            return result
        except TypeError:
            raise KeyError("no data has been loaded.")

    def free_memory(self):
        """Frees memory from the wrapped npz file."""
        _loaded_npz_files.free_memory(self._npz)


class AnalogReadoutLoop(AnalogReadoutABC):
    """
    Class for handling loop data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f": "freqs", "z": "z", "imbalance": ("calibration", structured_to_complex), "offset": "z_offset",
               "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "field": ("metadata", ("parameters", "field")), "temperature": ("metadata", analogreadout_temperature)}


class AnalogReadoutNoise(AnalogReadoutABC):
    """
    Class for handling noise data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    # "i_psd": ("psd", "I"), "q_psd": ("psd", "Q"), "f_psd": "f_psd" not using these from file but they are there
    CONVERT = {"f_bias": "freqs", "i_trace": ("noise", "I", np.real), "q_trace": ("noise", "Q", np.imag),
               "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}

    def __init__(self, npz_handle=None, channel=None, index=0):
        super().__init__(npz_handle=npz_handle, channel=channel, index=index)


class AnalogReadoutPulse(AnalogReadoutABC):
    """
    Class for handling pulse data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f_bias": "freqs", "i_trace": ("pulses", "I", np.real), "q_trace": ("pulses", "Q", np.imag),
               "offset": "zero", "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._energies = []

    def __getitem__(self, item):
        if item == 'energies':
            if self._energies:
                result = self._energies
            else:
                metadata = super().__getitem__("metadata")
                try:
                    laser_state = np.array(metadata['parameters']['laser'])
                    laser_state *= np.array([808, 920, 980, 1120, 1310])
                    laser_state = laser_state[laser_state != 0]
                    self._energies = tuple(ev_nm_convert(laser_state))
                except KeyError:
                    pass
                result = self._energies
        else:
            result = super().__getitem__(item)
        return result


def analogreadout_resonator(file_name, channel=None):
    """
    Class for loading in analogreadout resonator data.
    Args:
        file_name: string
            The resonator configuration file name.
        channel: integer
            The resonator channel for the data.
    Returns:
        loop_kwargs: list of dictionaries
            A list of keyword arguments to send to Loop.load().
    """
    directory = os.path.dirname(file_name)
    npz = np.load(file_name, allow_pickle=True)
    loop_kwargs = []
    for loop_name, parameters in npz['parameter_dict'].item().items():
        loop_file_name = os.path.join(directory, loop_name)
        if os.path.isfile(loop_file_name):
            loop_kwargs.append({"loop_file_name": loop_file_name, "channel": channel})
            if parameters['noise'][0]:
                n_noise = 1 + int(parameters['noise'][5])
                noise_name = "_".join(["noise", *loop_name.split("_")[1:]])
                noise_file_name = os.path.join(directory, noise_name)
                if os.path.isfile(noise_file_name):
                    noise_names = [noise_file_name] * n_noise
                    noise_kwargs = [{'index': ii} for ii in range(n_noise)]
                    loop_kwargs[-1].update({"noise_file_names": noise_names, "noise_kwargs": noise_kwargs,
                                            "channel": channel})
                else:
                    log.warning("Could not find '{}'".format(noise_file_name))
        else:
            log.warning("Could not find '{}'".format(loop_file_name))
    return loop_kwargs


class LegacyABC:
    """
    Abstract base class for handling data from the Legacy matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        index: tuple of integers (optional)
            An integer specifying which temperature and attenuation index to
            load. The default is None.
    """
    def __init__(self, config_file, channel, index=None):
        self.channel = channel
        self.index = index
        self._empty_fields = []
        self._do_not_clear = ['metadata']
        # load in data to the configuration file
        self._data = {'metadata': {}}
        config = loadmat(config_file, squeeze_me=True)
        for key in config.keys():
            if not key.startswith("_"):
                for name in config[key].dtype.names:
                    try:
                        self._data['metadata'][name] = float(config[key][name].item())
                    except ValueError:
                        self._data['metadata'][name] = config[key][name].item()
                    except TypeError:
                        try:
                            self._data['metadata'][name] = config[key][name].item().astype(float)
                        except ValueError:
                            self._data['metadata'][name] = config[key][name].item()  # spec_settings exception

    def __getitem__(self, item):
        value = self._data[item]
        if value is None and item not in self._empty_fields:
            self._load_data()
            value = self._data[item]
        return value

    def __getstate__(self):
        __dict__ = self.__dict__.copy()
        __dict__['_data'] = {}
        for key in self.__dict__['_data'].keys():
            if key not in self._do_not_clear:
                __dict__['_data'][key] = None
            else:
                __dict__['_data'][key] = self.__dict__['_data'][key]
        return __dict__

    def free_memory(self):
        """Frees memory from the wrapped data."""
        for key in self._data.keys():
            if key not in self._do_not_clear:
                self._data[key] = None

    def _load_data(self):
        raise NotImplementedError


class LegacyLoop(LegacyABC):
    """
    Class for handling loop data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load. The default is None
            which will raise an error forcing the user to directly specify the
            channel.
        index: tuple of integers
            An integer specifying which temperature and attenuation index to
            load. The default is None will raise an error forcing the user to
            directly specify the index.
    """
    def __init__(self, config_file, channel, index):
        super().__init__(config_file, channel, index=index)
        # load in the loop data
        time = os.path.basename(config_file).split('_')[2:]
        directory = os.path.dirname(config_file)
        mat_file = "sweep_data.mat" if not time else "sweep_data_" + "_".join(time)
        self._mat = os.path.join(directory, mat_file)
        self._empty_fields += ["imbalance"]
        self._data.update({"f": None, "z": None, "imbalance": None, "offset": None, "field": None, "temperature": None,
                           "attenuation": None})  # defer loading

    def _load_data(self):
        data = loadmat(self._mat, struct_as_record=False)['IQ_data'][0, 0]
        res = data.temps[0, self.index[0]].attens[0, self.index[1]].res[0, self.channel]
        self._data.update({"f": res.freqs.squeeze(), "z": res.z.squeeze(), "imbalance": None,
                           "offset": res.zeropt.squeeze(), "field": 0,
                           "temperature": data.temprange[0, self.index[0]] * 1e-3,
                           "attenuation": data.attenrange[0, self.index[1]]})


class LegacyNoise(LegacyABC):
    """
    Class for handling noise data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        index: tuple of integers (optional)
            An integer specifying which temperature and attenuation index to
            load. An additional third index may be included in the tuple to
            specify additional noise points. This is only needed if the data
            is from a resonator config.
        on_res: boolean (optional)
            A boolean specifying if the noise is on or off resonance. This is
            only used when the noise comes from the Sweep GUI. The default is
            True.
    """
    def __init__(self, config_file, channel, index=None, on_res=True):
        super().__init__(config_file, channel, index=index)
        # figure out the file specifics
        directory = os.path.dirname(os.path.abspath(config_file))
        self._sweep_gui = os.path.basename(config_file).split("_")[0] == "sweep"
        if self._sweep_gui:
            if self.index is None:
                raise ValueError("The index (temperature, attenuation) must be specified for Sweep GUI data.")
            temps = np.arange(self._data['metadata']['starttemp'],
                              self._data['metadata']['stoptemp'] + self._data['metadata']['steptemp'] / 2,
                              self._data['metadata']['steptemp'])
            attens = np.arange(self._data['metadata']['startatten'],
                               self._data['metadata']['stopatten'] + self._data['metadata']['stepatten'] / 2,
                               self._data['metadata']['stepatten'])
            label = "a" if on_res else "b"
            label += "{:g}".format(index[2]) + "-" if len(index) > 2 and index[2] != 0 else "-"
            file_name = ("{:g}".format(temps[index[0]]) + "-" + "{:g}".format(channel // 2 + 1) + label +
                         "{:g}".format(attens[index[1]]) + ".ns")
            n_points = (self._data['metadata']['adtime'] * self._data['metadata']['noiserate'] /
                        self._data['metadata']['decfac'])
            self._data['attenuation'] = attens[index[1]]
            self._data['sample_rate'] = self._data['metadata']['noiserate']
        else:
            time = os.path.basename(config_file).split('.')[0].split('_')[2:]
            file_name = "pulse_data.ns" if not time else "pulse_data_" + "_".join(time) + ".ns"
            n_points = self._data['metadata']['noise_adtime'] * self._data['metadata']['samprate']
            self._data['attenuation'] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
            self._data['sample_rate'] = self._data['metadata']["samprate"]
        self._do_not_clear += ['attenuation', 'sample_rate']
        # load the data
        assert n_points.is_integer(), "The noise adtime and sample rate do not give an integer number of data points"
        self._n_points = int(n_points)
        self._bin = os.path.join(directory, file_name)
        self._data.update({"i_trace": None, "q_trace": None, "f_bias": None})  # defer loading

    def _load_data(self):
        # % 2 for resonator data
        i_trace, q_trace, f = load_legacy_binary_data(self._bin, self.channel % 2, self._n_points)
        self._data.update({"i_trace": i_trace, "q_trace": q_trace, 'f_bias': f})


class LegacyPulse(LegacyABC):
    """
    Class for handling pulse data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        energies: number or iterable of numbers (optional)
            The known energies in the pulse data. The default is an empty
            tuple.
        wavelengths number or iterable of numbers (optional)
            If energies is not specified, wavelengths can be specified instead
            which are internally converted to energies. The default is an empty
            tuple.
    """
    def __init__(self, config_file, channel, energies=(), wavelengths=()):
        channel = channel % 2  # channels can't be > 1
        super().__init__(config_file, channel=channel)
        # record the photon energies
        if energies != ():
            self._data["energies"] = tuple(np.atleast_1d(energies))
        elif wavelengths != ():
            self._data["energies"] = tuple(ev_nm_convert(np.atleast_1d(wavelengths)))
        else:
            self._data["energies"] = ()
        # get the important parameters from the metadata
        self._data["f_bias"] = self._data['metadata']["f0" + "{:g}".format(channel + 1)]
        self._data["offset"] = None
        self._data["attenuation"] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
        self._data['sample_rate'] = self._data['metadata']["samprate"]
        self._do_not_clear += ['f_bias', 'attenuation', 'offset', 'energies', 'sample_rate']
        self._empty_fields += ["offset"]

        directory = os.path.dirname(os.path.abspath(config_file))
        time = os.path.basename(config_file).split('.')[0].split('_')[2:]
        file_name = "pulse_data.dat" if not time else "pulse_data_" + "_".join(time) + ".dat"
        self._bin = os.path.join(directory, file_name)
        self._n_points = int(self._data['metadata']['numpts'])
        self._data.update({"i_trace": None, "q_trace": None})  # defer loading

    def _load_data(self):
        i_trace, q_trace, _ = load_legacy_binary_data(self._bin, self.channel, self._n_points, noise=False)
        self._data.update({"i_trace": i_trace, "q_trace": q_trace})


def legacy_resonator(config_file, channel=None, noise=True):
    """
    Class for loading in legacy matlab resonator data.
    Args:
        config_file: string
            The resonator configuration file name.
        channel: integer
            The resonator channel for the data.
        noise: boolean
            If False, ignore the noise data. The default is True.
    Returns:
        loop_kwargs: list of dictionaries
            A list of keyword arguments to send to Loop.load().
    """
    directory = os.path.dirname(config_file)
    config = loadmat(config_file, squeeze_me=True)['curr_config']
    temperatures = np.arange(config['starttemp'].astype(float),
                             config['stoptemp'].astype(float) + config['steptemp'].astype(float) / 2,
                             config['steptemp'].astype(float))
    attenuations = np.arange(config['startatten'].astype(float),
                             config['stopatten'].astype(float) + config['stepatten'].astype(float) / 2,
                             config['stepatten'].astype(float))

    loop_kwargs = []
    for t_index, temp in enumerate(temperatures):
        for a_index, atten in enumerate(attenuations):
            loop_kwargs.append({"loop_file_name": config_file, "index": (t_index, a_index), "data": LegacyLoop,
                                "channel": channel})
            if config['donoise'] and noise:
                group = channel // 2 + 1
                # on resonance file names
                on_res = glob.glob(os.path.join(directory, "{:g}-{:d}a*-{:g}.ns".format(temp, group, atten)))
                noise_kwargs = []
                for file_name in on_res:
                    # collect the index for the file name
                    base_name = os.path.basename(file_name)
                    index2 = base_name.split("a")[1].split("-")[0]
                    index = (t_index, a_index, int(index2)) if index2 else (t_index, a_index)
                    noise_kwargs.append({"index": index, "on_res": True, "data": LegacyNoise, "channel": channel})
                # off resonance file names
                off_res_names = glob.glob(os.path.join(directory, "{:g}-{:d}b*-{:g}.ns".format(temp, group, atten)))
                for file_name in off_res_names:
                    # collect the index for the file name
                    base_name = os.path.basename(file_name)
                    index2 = base_name.split("b")[1].split("-")[0]
                    index = (t_index, a_index, int(index2)) if index2 else (t_index, a_index)
                    noise_kwargs.append({"index": index, "on_res": False, "data": LegacyNoise, "channel": channel})
                loop_kwargs[-1].update({"noise_file_names": [config_file] * len(on_res + off_res_names),
                                        "noise_kwargs": noise_kwargs})
                if not noise_kwargs:
                    log.warning("Could not find noise files for '{}'".format(config_file))
    return loop_kwargs
