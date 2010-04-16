#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction
from scikits.audiolab import play, wavwrite

from envelope import Exponential
from oscillator import Osc

np.set_printoptions(precision=4, suppress=True)

class Harmonic:
    """Harmonical overtones"""
    
    def __init__(self, func=lambda x: 1+x, n=8):
        # Set overtones
        self.func = func
        self.limit = n
        self.overtones = np.array(map(func, np.arange(0, n, dtype=float)))
        # Set envelopes for overtones
    
    def sample(self, freq, iter):
        oscs = Osc.freq(freq) * self.overtones
        oscs = np.ma.masked_array(oscs, np.equal(oscs, Osc(0, 1)), None).compressed()
        frames = np.zeros(len(iter), dtype=complex)
        for o in oscs:
            # e = Exponential(0, amp=float(freq)/o.frequency*float(freq)) # square waves
            # e = Exponential(0, amp=float(freq)**2/o.frequency**2*float(freq)) # triangle waves
            e = Exponential(-o.frequency/100.0) # sine waves
            frames += o[iter] * e[iter]
        return frames / max( abs(max(frames)), len(oscs), 1.0 )
    
    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        # convert time to frames
        return self.sample(item)