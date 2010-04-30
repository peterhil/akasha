#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from fractions import Fraction
from scikits.audiolab import play, wavwrite

from envelope import Exponential
from oscillator import Osc

# np.set_printoptions(precision=4, suppress=True)

class Sound(defaultdict):
    """Sound groups."""

    def __init__ (self):
        defaultdict.__init__(self, object)

    def sample(self, *args):
        '''Pass parameters to all sound objects and update states.'''
        for sndobj in self:
            frames = sndobj.sample(*args)
            self[sndobj] = frames

    def add(self, sndobj, offset=0):
        '''Add a new sndobj to self.'''
        self[sndobj] = None

    def stat(self):
        '''Return a tuple containing the state of each sound object.'''
        return tuple(self.values())

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)
