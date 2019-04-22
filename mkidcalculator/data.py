import os
import logging
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_loaded_npz_files = {}  # cache of already loaded files


class AnalogReadoutABC:
    def __init__(self, npz_handle=None, channel=None, index=None):
        self.channel = channel
        self.index = index
        global _loaded_npz_files
        # if string load with numpy and save if it hasn't been loaded before
        if isinstance(npz_handle, str):
            npz_handle = os.path.abspath(npz_handle)
            # check if already loaded
            if npz_handle in _loaded_npz_files.keys():
                self._npz = _loaded_npz_files[npz_handle]
                log.info("loaded from cache: {}".format(npz_handle))
            else:
                npz = np.load(npz_handle)
                self._npz = npz
                log.info("loaded: {}".format(npz_handle))
                _loaded_npz_files[npz_handle] = npz
                log.info("saved to cache: {}".format(npz_handle))
        # if NpzFile skip loading but save if it hasn't been loaded before
        elif isinstance(npz_handle, np.lib.npyio.NpzFile):
            self._npz = npz_handle
            file_name = os.path.abspath(npz_handle.fid.name)
            if file_name not in _loaded_npz_files.keys():
                _loaded_npz_files[file_name] = npz_handle
                log.info("saved to cache: {}".format(file_name))
        # allow for dummy object creation
        elif npz_handle is None:
            self._npz = npz_handle
        else:
            raise ValueError("'npz_handle' must be a valid file name or a numpy npz file object.")

    def __getitem__(self, item):
        # get conversion values
        try:
            convert = self.CONVERT[item]
        except KeyError:
            raise KeyError("allowed keys for this data structure are in {}".format(list(self.CONVERT.keys())))
        try:
            # get the result from the npz file
            result = self._npz[convert[0] if isinstance(convert, tuple) else convert]
            if result.dtype == np.dtype('O'):
                # if it's an object unpack it
                result = result.item()
            else:
                # else get the channel and index
                if self.channel is not None:
                    result = result[self.channel]
                if self.index is not None:
                    result = result[self.index]
            # more conversion
            if isinstance(convert, tuple) and len(convert) > 1:
                if not callable(convert[1]):
                    try:
                        # grab another index
                        result = result[convert[1]]
                    except IndexError:
                        # if that failed run a function (for complex formatted data)
                        result = convert[2](result)
                else:
                    result = convert[1](result)
            return result
        except TypeError:
            raise KeyError("no data has been loaded.")


class AnalogReadoutLoop(AnalogReadoutABC):
    CONVERT = {"f": "freqs", "z": "z", "imbalance": "calibration", "offset": "z_offset", "metadata": "metadata"}


class AnalogReadoutNoise(AnalogReadoutABC):
    CONVERT = {"f_bias": "freqs", "i_trace": ("noise", "I", np.real), "q_trace": ("noise", "Q", np.imag),
               "metadata": "metadata"}
    # "i_psd": ("psd", "I"), "q_psd": ("psd", "Q"), "f_psd": "f_psd" not using these from file but they are there


class AnalogReadoutPulse(AnalogReadoutABC):
    CONVERT = {"f_bias": "freqs", "i_trace": ("pulses", "I", np.real), "q_trace": ("pulses", "Q", np.imag),
               "offset": "zero", "metadata": "metadata"}

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
                    self._energies = tuple(1239.842 / laser_state)  # 1239.842 nm eV = h c
                except KeyError:
                    pass
                result = self._energies
        else:
            result = super().__getitem__(item)
        return result
