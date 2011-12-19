#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction
from cmath import rect, pi
from copy import copy, deepcopy

from audio.envelope import Exponential
from audio.oscillator import Osc, Frequency
from audio.generators import Generator
from timing import Sampler

from utils.decorators import memoized
from utils.math import random_phase


class Overtones(object, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(self, sndobj=Osc.freq(220.0), n=8, func=lambda x: 1+x, damping=None, rand_phase=False):
        self.base = sndobj
        self.n = n
        self.func = func
        self.damping = damping or (lambda f, a=1.0: (-np.log2(f)/(5.0), min(1.0, a*(self.frequency/f))))   # Sine waves
        self.rand_phase = rand_phase

    @property
    def frequency(self):
        return self.base.frequency

    @frequency.setter
    def frequency(self, f):
        self.base.frequency = f  # Use Trellis, and make a interface for frequencies

    @property
    def max_overtones(self):
        if self.frequency == 0:
            return 1
        return int(Sampler.rate / (2.0 * self.frequency))

    @property
    def limit(self):
        return min(self.max_overtones, self.n)

    @property
    def overtones(self):
        return np.apply_along_axis(self.func, 0, np.arange(0, self.limit, dtype=np.float32))

    @property
    def oscs(self):
        # TODO cleanup - make an interface for different Oscs!
        return self.gen_oscs(self.frequency, self.overtones)

    def gen_oscs2(self, *args):
        return map(lambda f: self.base.__class__(f, *args), self.frequency * self.overtones)

    def oscs2(self):
        base = self.base.__class__
        if base.__name__ == 'Super':
            #oscs = map(base, self.frequency * self.overtones, [list(self.base.superness)] * self.limit)
            oscs = self.gen_oscs2(self.base.superness)
            # for o in oscs:
            #     o.superness = self.base.superness
        else:
            oscs = map(base, self.frequency * self.overtones)
        return oscs

    def sample(self, iter):
        frames = np.zeros(len(iter), dtype=complex)
        
        # for o in self.oscs2():
        for f in self.overtones:
            o = copy(self.base)     # Uses deepcopy to preserve superness & other attrs
            o.frequency = Frequency(f * self.frequency)
            if o.frequency == 0: break
            
            # e = Exponential(0, amp=float(self.frequency/o.frequency*float(self.frequency) # square waves
            # e = Exponential(0, amp=float(self.frequency**2/o.frequency**2*float(self.frequency) # triangle waves
            # e = Exponential(-o.frequency/100.0) # sine waves
            e = Exponential(self.damping(o.frequency)) # sine waves
            
            if self.rand_phase:
                frames += o[iter] * random_phase() * e[iter]    # Move phases to Osc/Frequency!!!
            else:
                frames += o[iter] * e[iter]
        
        return frames #/ max( abs(max(frames)), 1.0 )

    def __repr__(self):
        return "<Overtones(%s): frequency = %s, overtones = %s, limit = %s, func = %s, damping = %s>" % \
                (self.base, self.frequency, self.overtones, self.limit, self.func, self.damping)

