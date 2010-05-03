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
        self.sounds = defaultdict.__init__(self, object)

    def sample(self, iter):
        '''Pass parameters to all sound objects and update states.'''
        sound = np.zeros(len(iter), dtype=complex)
        for sndobj in self.sounds:
            sound += sndobj.sample(iter)
        return sound / max( abs(max(sound)), len(sound), 1.0 )

    def add(self, sndobj, offset=0):
        '''Add a new sndobj to self.'''
        self.sounds[sndobj] = sndobj

    def stat(self):
        '''Return a tuple containing the state of each sound object.'''
        return tuple(self.sounds.values())

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)
