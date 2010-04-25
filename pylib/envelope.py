#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi, exp
import math
import numpy as np
from oscillator import Sampler

class Exponential:
    """Exponential decay and growth for envelopes."""
    
    def __init__(self, decay, amp=1.0):
        self.decay = decay
        self.amp = amp
    
    def exponential(self, iter):
        frames = np.array(iter) / float(Sampler.rate)
        return self.amp * np.exp(self.decay * frames)
    
    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        # convert time to frames
        return self.exponential(item)
    
    def __repr__(self):
        return "Exponential(%s, %s)" % (self.decay, self.amp)
    
    def __str__(self):
        return "<Exponential: decay=%s, amp=%s>" % (self.decay, self.amp)

    