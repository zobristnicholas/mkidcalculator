import os
import pickle
import logging

from mkidcalculator.io.resonator import Resonator
from mkidcalculator.io.data import analogreadout_sweep

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep:
    """A class for organizing data from multiple resonators."""
    def __init__(self):
        self.resonators = []

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
        log.info("saved sweep as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Sweep class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            sweep = pickle.load(f)
        assert isinstance(sweep, cls), "'{}' does not contain a Sweep class.".format(file_name)
        log.info("loaded sweep from '{}'".format(file_name))
        return sweep

    def add_resonators(self, resonators):
        """
        Add Resonator objects to the sweep.
        Args:
            resonators: Resonator class or iterable of Resonator classes
                The resonators that are to be added to the Sweep.
        """
        if isinstance(resonators, Resonator):
            resonators = [resonators]
        # append resonator data
        for resonator in resonators:
            resonator.sweep = self
            self.resonators.append(resonator)

    def remove_resonators(self, indices):
        """
        Remove resonators from the sweep.
        Args:
            indices: integer or iterable of integers
                The indices in sweep.resonators that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.resonators.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Resonator objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for resonator in self.resonators:
            resonator.free_memory(directory=directory)

    @classmethod
    def from_file(cls, sweep_file_name, data=analogreadout_sweep, sort=True, **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the resonator
        data loaded.
        Args:
            sweep_file_name: string
                The file name for the sweep data.
            data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to
                Resonator.from_file().
            sort: boolean (optional)
                Sort the loop data in each resonator by its power, field, and
                temperature. Also sort noise data and pulse data lists for each
                loop by their bias frequencies. The default is True. If False,
                the input order is preserved. The resonators list is not
                sorted.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            sweep: object
                A Sweep() object containing the loaded data.
        """
        sweep = cls()
        res_kwarg_list = data(sweep_file_name, **kwargs)
        resonators = []
        for kws in res_kwarg_list:
            kws.update({"sort": sort})
            resonators.append(Resonator.from_file(**kws))
        sweep.add_resonators(resonators)
        return sweep

    def _set_directory(self, directory):
        self._directory = directory
        for resonator in self.resonators:
            resonator._set_directory(self._directory)
