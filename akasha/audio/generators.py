#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Sound generating objects’ base classes.
"""

import numpy as np

from akasha.funct import blockwise
from akasha.timing import sampler


class Generator(object):
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
                res = np.clip(np.array([item.start, item.stop]), a_min=None, a_max=len(self))
                for i in xrange(len(res)):
                    if res[i] is not None and np.sign(res[i]) == -1:
                        res[i] = max(-len(self) - 1, res[i])
                item = slice(res[0], res[1], item.step)
                item = np.arange(*(item.indices(item.stop or (len(self) - 1))))
            else:
                item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)

    def at(self, t):
        """
        Sample sound generator at times (t).
        """
        raise NotImplementedError("Please implement method at() in a subclass.")

    def sample(self, frames):
        """
        Sample the sound at frame numbers.
        """
        if isinstance(frames, slice):
            frames = np.arange(*frames.indices(frames.stop))

        if isinstance(frames, np.ndarray):
            return self.at(np.asarray(frames, dtype=np.int64) / float(sampler.rate))
        elif np.iterable(frames):
            return self.at(np.fromiter(frames, dtype=np.int64) / float(sampler.rate))
        elif np.isreal(frames):
            return self.at(frames / float(sampler.rate))
        else:
            raise NotImplementedError(
                "Sampling generators with objects of type %s not implemented yet." % type(frames)
            )

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
            step = ((item.step or 1) % self.period or 1)
            start = ((item.start or 0) % self.period)
            element_count = abs((item.stop or self.period) - (item.start or 0))
            stop = start + (element_count * step)
            item = np.arange(*(slice(start, stop, step).indices(stop)))
        if np.isscalar(item):
            return self.sample[np.array(item, dtype=np.int64) % self.period]
        else:
            return self.sample[np.fromiter(item, dtype=np.int64) % self.period]

    def at(self, t):
        """
        Sample {} at times (t).
        """.format(self.__class__.__name__)
        return self.sample(t % self.period)

    # Disabled because Numpy gets clever (and slow) when a sound objects have length and
    # they're made into an object array...
    # def __len__(self):
    #     return self.period
