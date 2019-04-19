import pandas as pd


class Sweep:
    def __init__(self):
        self.loops = []

    def save(self):
        pass

    @classmethod
    def load(cls):
        raise NotImplementedError
