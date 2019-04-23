import pandas as pd
from operator import itemgetter

from mkidcalculator.loop import Loop
from mkidcalculator.data import analogreadout_sweep


class Sweep:
    def __init__(self):
        self.loops = []
        self.temperatures = []
        self.powers = []
        self.fields = []

    def save(self):
        raise NotImplementedError

    def add_loops(self, loops, sort=True):
        """
        Add loop data sets to the sweep.
        Args:
            loops: iterable of Loop() classes
                The loop data sets that are to be added to the Sweep.
            sort: boolean (optional)
                Sort the loop data list by its power, field, and temperature.
                The default is True. If False, the order of the loop data sets
                is preserved.
        """
        # append loop data
        for loop in loops:
            self.loops.append(loop)
            self.powers.append(loop.power)
            self.fields.append(loop.field)
            self.temperatures.append(loop.temperature)
        # sort
        if sort:
            lp = zip(*sorted(zip(self.powers, self.fields, self.temperatures, self.loops), key=itemgetter(0, 1, 2)))
            self.powers, self.fields, self.temperatures, self.loops = (list(t) for t in lp)

    @classmethod
    def load(cls, sweep_file_name, sweep_data=analogreadout_sweep, sort=True, **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the loop, noise
        and pulse data loaded.
        Args:
            sweep_file_name: string
                The file name for the sweep data. Typically it is a
                configuration file with all the information needed to load in
                the data.
            sweep_data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to Loop.load().
            sort: boolean (optional)
                Sort the loop data by its power, field, and temperature. Also
                sort noise data and pulse data lists by their bias frequency.
                The default is True. If False, the input order is preserved.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to Loop.load() not
                specified by sweep_data.
        """
        # create sweep
        sweep = cls()
        # load loop kwargs based on the sweep file
        loop_kwargs_list = sweep_data(sweep_file_name)
        loops = []
        # load loops
        for kws in loop_kwargs_list:
            kws.update(kwargs)
            kws.update({"sort": sort})
            loops.append(Loop.load(**kws))
        sweep.add_loops(loops, sort=sort)
        return sweep

    def lmfit(self):
        raise NotImplementedError

    def emcee(self):
        raise NotImplementedError
