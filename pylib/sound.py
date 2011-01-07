#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import hilbert

from collections import defaultdict
from fractions import Fraction
from numbers import Number

# My modules
from envelope import Exponential
from oscillator import Osc
from harmonics import Harmonic
from generators import Generator
from utils import play, write, read
from utils.graphing import *
from utils.animation import *

# np.set_printoptions(precision=4, suppress=True)

class Sound(object, Generator):
    """Sound groups."""
    
    def __init__ (self, *args):
        self.sounds = {}
        for s in args:
            self.add(s)
        
    def sample(self, iter):
        """Pass parameters to all sound objects and update states."""
        if isinstance(iter, Number):
            # FIXME should return scalar, not array!
            start = int(iter)
            stop = start + 1
        else:
            start = iter.start or 0
            stop = iter.stop
        sl = (start, stop)
        
        sound = np.zeros((stop - start), dtype=complex)
        for sl in self.sounds:
            print "Slice start %s, stop %s" % sl
            for sndobj in self.sounds[sl]:
                print "Sound object %s" % sndobj
                sound += sndobj[iter]
        return sound / max( len(self), 1.0 )

    def __len__(self):
        return sum(map(lambda s: len(self.sounds[s]), self.sounds))

    # def __repr__(self):
    #     components = u''
    #     for obj in self.sounds:
    #         components += repr(obj)
    #     return "Sound(%s)" % (components)
    
    # Use s.sounds.append(sndobj)
    # s.sounds.index(sndobj)
    
    def add(self, sndobj, start=0, dur=None):
        """Add a new sndobj to self."""
        if dur:
            end = start + dur
        elif hasattr(sndobj, "len"):
            end = start + len(sndobj)
        else:
            end = None
        
        sl = (start, end)

        if (self.sounds.has_key(sl)):
            self.sounds[sl].append(sndobj)
        else:
            self.sounds[sl] = [sndobj]
        return self

    # def stat(self):
    #     '''Return a tuple containing the state of each sound object.'''
    #     return tuple(self.sounds.values())
