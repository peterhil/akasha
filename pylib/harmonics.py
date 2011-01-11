#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction

from envelope import Exponential
from oscillator import Osc
from generators import Generator
from utils import play, write

# np.set_printoptions(precision=4, suppress=True)


class Harmonic(object, Generator):
    """Harmonical overtones"""

    def __init__(self, freq=220, func=lambda x: 1+x, damping=None, n=8, superness=2):
        # Set overtones
        self.freq = freq
        self.func = func
        self.limit = n
        self.superness = superness
        self.damping = damping or (lambda f, a=1.0: (-f/100.0, a/(f/self.freq)))   # Sine waves
        if n <= 20:
            self.overtones = np.array(map(func, np.arange(0, n)), dtype=np.float32)
        else:
            # numpy.apply_along_axis is faster than map for larger n
            self.overtones = np.apply_along_axis(func, 0, np.arange(0, n, dtype=np.float32))

    def __call__(self, freq):
        self.freq = freq

    def sample(self, iter):
        oscs = Osc.freq(self.freq, self.superness) * self.overtones
        # oscs = np.ma.masked_array(oscs, np.equal(oscs, Osc(0)), None).compressed()
        oscs = filter(lambda x: x!=Osc(0), oscs)  # Quick hack to prevent problems with numpy broadcasting and new style classes
        frames = np.zeros(len(iter), dtype=complex)
        for o in oscs:
            # e = Exponential(0, amp=float(self.freq)/o.frequency*float(self.freq)) # square waves
            # e = Exponential(0, amp=float(self.freq)**2/o.frequency**2*float(self.freq)) # triangle waves
            # e = Exponential(-o.frequency/100.0) # sine waves
            e = Exponential(self.damping(o.frequency)) # sine waves
            frames += o[iter] * e[iter]
        return frames / max( abs(max(frames)), len(oscs), 1.0 )
    
    def __repr__(self):
        if hasattr(self, 'freq'):
            freq = self.freq
        else:
            freq = None
        return "<Harmonic(%s): superness = %s;\n\tovertones = %s;limit = %s;\n\tfunc = %s;\n\tdamping = %s>" % \
                (freq, self.superness, self.overtones, self.limit, self.func, self.damping)

