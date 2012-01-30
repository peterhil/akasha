#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy as sp
from cmath import rect, polar, phase, pi, exp
from fractions import Fraction

from audio.generators import Generator
from timing import Sampler
from utils.math import minfloat, maxfloat

class Exponential(object, Generator):
    """Exponential decay and growth for envelopes."""

    def __init__(self, rate=0.0, amp=1.0, *args, **kwargs):
        if isinstance(rate, tuple):
            self.rate, self.amp = rate
        else:
            self.rate = rate
            self.amp = amp

    @property
    def half_life(self):
        if self.rate == 0:
            return np.inf
        else:
            return math.log(2.0) / -self.rate * Sampler.rate

    @property
    def zero_point(self):
        """
        Returns the time required to reach zero from starting amplitude.
        For e[e.zero_point+offset] to be zero, offset = -1 for growth (positive rate) and +1 for decay (negative rate).
        """
        return self.half_life * (minfloat(self.amp)[1]+1)

    def sample(self, iterable):
        # Convert frame numbers to time (ie. 44100 => 1.0)
        frames = np.array(iterable) / float(Sampler.rate)
        return self.amp * np.exp(self.rate * frames)

    def __len__(self):
        return int(math.ceil(np.abs(self.zero_point)))

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.rate, self.amp)

    def __str__(self):
        return "<%s: rate=%s, amp=%s>" % (self.__class__.__name__, self.rate, self.amp)


class Attack(Exponential):
    """Exponential attack (reversed decay/growth) envelope"""

    def __init__(self, rate=0.0, amp=1.0, *args, **kwargs):
        super(Attack, self).__init__(rate, amp, *args, **kwargs)

    def sample(self, iter, threshold=1.0e-6):
        orig_length = len(iter)
        frames = np.zeros(orig_length)

        atck = super(Attack, self).sample(iter)
        atck = filter(lambda x: x > threshold, atck)[::-1] # filter silence and reverse
        sus_level = atck[-1]
        frames.fill(sus_level)
        frames[:len(atck)] = atck
        del(atck)
        return frames


class Gamma(object, Generator):
    """Gamma cumulative distribution function derived envelope."""

    def __init__(self, shape=1.0, scale=1.0):
        if isinstance(shape, tuple):
            self.shape, self.scale = shape
        else:
            self.shape = shape
            self.scale = scale  # Inverse rate

    def sample(self, iterable):
        frames = (np.array(iterable) / float(Sampler.rate)) * (1.0/max(self.scale, 1e-06)) # Make scale into rate
        return sp.special.gammaincc(self.shape, frames)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.shape, self.scale)

    def __str__(self):
        return "<%s: shape=%s, scale=%s>" % (self.__class__.__name__, self.shape, self.scale)


class Timbre(object, Generator):
    """Defines an envelope timbre for frequencies."""

    def __init__(self):
        pass

