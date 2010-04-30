#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class Generator:

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)


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
            step = ((item.step or 1) % len(self) or 1)
            start = ((item.start or 0) % len(self))
            element_count = abs((item.stop or len(self)) - (item.start or 0))
            stop = start + (element_count * step)
            # Construct an array of indices.
            item = np.arange(*(slice(start, stop, step).indices(stop)))
            # print item[-1] % self.period # Could be used as cursor
        return self.samples[np.array(item) % len(self)]

    def __len__(self):
        return self.period