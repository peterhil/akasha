#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
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
        return math.log(2.0) / -self.rate * Sampler.rate

    def sample(self, iterable):
        # Convert frame numbers to time (ie. 44100 => 1.0)
        frames = np.array(iterable) / float(Sampler.rate)
        return self.amp * np.exp(self.rate * frames)

    def __len__(self):
        if self.rate == 0:
            return np.inf
        elif self.rate < 0:
            # Exponential decay approaching zero
            return int(math.ceil(self.half_life*(minfloat(self.amp)[1]+1)))
        else:
            # Exponential growth approaching inf
            return int(math.ceil(-self.half_life*(minfloat(self.amp)[1]+1)))

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


class Timbre(object, Generator):
    """Defines an envelope timbre for frequencies."""

    def __init__(self):
        pass

