"""
Sound generating objects’ base classes.
"""

import numpy as np

from akasha.funct.itertools import blockwise
from akasha.timing import sampler
from akasha.utils.python import class_name


class Generator:
    """
    Sound generator base class.
    """

    def __getitem__(self, item):
        """
        Slicing support.
        """
        if isinstance(item, slice):
            if hasattr(self, '__len__'):
                # Mimick ndarray slicing, ie. clip the slice indices
                res = np.clip(
                    np.array([item.start, item.stop]),
                    a_min=None,
                    a_max=len(self),
                )
                for i, index in enumerate(res):
                    if index is not None and np.sign(index) == -1:
                        res[i] = max(-len(self) - 1, index)
                item = slice(res[0], res[1], item.step)
                stop = item.stop or (len(self) - 1)
                item = np.arange(*(item.indices(stop)))
            else:
                item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)

    def at(self, t):
        """
        Sample sound generator at times (t).
        """
        raise NotImplementedError(
            "Please implement method at() in a subclass."
        )

    def sample(self, frames):
        """
        Sample the sound at frame numbers.
        """
        if isinstance(frames, slice):
            frames = np.arange(*frames.indices(frames.stop))

        fs = float(sampler.rate)
        if isinstance(frames, np.ndarray):
            times = np.asarray(frames, dtype=np.int64) / fs
        elif np.iterable(frames):
            times = np.fromiter(frames, dtype=np.int64) / fs
        elif np.isreal(frames):
            times = frames / fs
        else:
            raise NotImplementedError(
                f"Sampling generators with type '{type(frames)}' "
                "not implemented yet."
            )

        return self.at(times)

    def __iter__(self):
        return blockwise(self, sampler.blocksize())


class PeriodicGenerator(Generator):
    """
    Sound objects with some repeating period.
    """

    def __getitem__(self, item):
        """
        Slicing support.

        If given a slice the behaviour will be:

        - Step defaults to 1, is wrapped modulo period, and can't be zero!
        - Start defaults to 0, is wrapped modulo period
        - Number of elements returned is the absolute differerence of
          stop - start (or period and 0 if either value is missing)

        Element count is multiplied with step to produce the same
        number of elements for different step values.
        """
        if isinstance(item, slice):
            step = (item.step or 1) % self.period or 1
            start = (item.start or 0) % self.period
            element_count = abs(
                (item.stop or self.period) - (item.start or 0)
            )
            stop = start + (element_count * step)
            item = np.arange(*(slice(start, stop, step).indices(stop)))

        if np.isscalar(item):
            return self.cycle[np.array(item, dtype=np.int64) % self.period]
        return self.cycle[np.fromiter(item, dtype=np.int64) % self.period]

    def at(self, t):
        f"""Sample {class_name(self)} at times (t)."""
        return self.sample(t % self.period)

    # Disabled because Numpy gets clever (and slow) when a sound objects
    # have length and they're made into an object array...
    # def __len__(self):
    #     return self.period
