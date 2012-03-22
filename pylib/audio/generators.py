#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

from sys import maxint

from ..control.io import audio
from ..funct import blockwise, blockwise2
from ..timing import Sampler


class Generator:
    # Could be new style class, but this causes many problems because numpy
    # uses obj[0] (arrays first element) to determine it's type and then
    # automatically broadcasts oscs to their complex samples!
    # This could maybe be prevented by using custom __get* methods or descriptors.

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
           # Construct an array of indices.
           item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)

    def __call__(self, slice):
        return self.__getitem__(slice)

    def __iter__(self):
        return blockwise2(self, 1, Sampler.blocksize()) # FIXME start should be 0, fix blockwise2!

    def next(self):
        it = iter(self)
        while True:
            try:
                yield it.next()
            except StopIteration:
                break
        it.close()

    def play(self, *args, **kwargs):
        audio.play(self, *args, **kwargs)


class PeriodicGenerator(Generator):

    def __getitem__(self, item):
        """Slicing support. If given a slice the behaviour will be:

        # Step defaults to 1, is wrapped modulo period, and can't be zero!
        # Start defaults to 0, is wrapped modulo period
        # Number of elements returned is the absolute differerence of
        # stop - start (or period and 0 if either value is missing)
        # Element count is multiplied with step to produce the same
        # number of elements for different step values.
        """
        if isinstance(item, slice):
            step = ((item.step or 1) % self.period or 1)
            start = ((item.start or 0) % self.period)
            element_count = abs((item.stop or self.period) - (item.start or 0))
            stop = start + (element_count * step)
            # Construct an array of indices.
            item = np.arange(*(slice(start, stop, step).indices(stop)))
        return self.sample[np.array(item) % self.period]

    # Disabled because Numpy gets clever (and slow) when a sound objects have length and
    # they're made into an object array...
    # def __len__(self):
    #     #return maxint
    #     return self.period
