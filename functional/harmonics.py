#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction
from scikits.audiolab import play, wavwrite

from envelope import Exponential
from oscillator import Osc
from generators import Generator

# np.set_printoptions(precision=4, suppress=True)


class Harmonic(Generator):
    """Harmonical overtones"""

    def __init__(self, func=lambda x: 1+x, n=8):
        # Set overtones
        self.func = func
        self.limit = n
        self.overtones = np.array(map(func, np.arange(0, n, dtype=float)))

    def __call__(self, freq):
        self.freq = freq
        return self

    def sample(self, iter):
        oscs = Osc.freq(self.freq) * self.overtones
        oscs = np.ma.masked_array(oscs, np.equal(oscs, Osc(0, 1)), None).compressed()
        frames = np.zeros(len(iter), dtype=complex)
        for o in oscs:
            # e = Exponential(0, amp=float(self.freq)/o.frequency*float(self.freq)) # square waves
            # e = Exponential(0, amp=float(self.freq)**2/o.frequency**2*float(self.freq)) # triangle waves
            e = Exponential(-o.frequency/100.0) # sine waves
            frames += o[iter] * e[iter]
        return frames / max( abs(max(frames)), len(oscs), 1.0 )
