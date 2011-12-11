#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction
from cmath import rect, pi
from copy import deepcopy

from audio.envelope import Exponential
from audio.oscillator import Osc, Frequency
from audio.generators import Generator

from utils.math import random_phase
from timing import Sampler


class Overtones(object, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(self, sndobj=Osc.freq(220.0), n=8, func=lambda x: 1+x, damping=None, rand_phase=False):
        self.base = sndobj
        self.n = n
        self.func = func
        self.damping = damping or (lambda f, a=1.0: (-f/100.0, a/(f/self.frequency)))   # Sine waves
        self.rand_phase = rand_phase

    @property
    def frequency(self):
        return self.base.frequency

    @frequency.setter
    def frequency(self, f):
        self.base.frequency = f  # Use Trellis, and make a interface for frequencies

    @property
    def max_overtones(self):
        return int(Sampler.rate / (2.0 * self.frequency))

    @property
    def limit(self):
        return min(self.max_overtones, self.n)

    @property
    def overtones(self):
        return np.apply_along_axis(self.func, 0, np.arange(0, self.limit, dtype=np.float32))

    def sample(self, iter):
        frames = np.zeros(len(iter), dtype=complex)
        
        for f in self.overtones:
            o = deepcopy(self.base)     # Uses deepcopy to preserve superness & other attrs
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
        
        return frames / max( abs(max(frames)), self.limit, 1.0 )

    def __repr__(self):
        return "<Overtones(%s): frequency = %s, overtones = %s, limit = %s, func = %s, damping = %s>" % \
                (self.base, self.frequency, self.overtones, self.limit, self.func, self.damping)

