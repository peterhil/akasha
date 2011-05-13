#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from cmath import rect, polar, phase, pi, exp
from fractions import Fraction
# My modules
from generators import Generator
from timing import Sampler


# Following two methods are modified from:
# http://seun-python.blogspot.com/2009/06/floating-point-min-max.html

def minfloat(guess):
    i = 0
    while(guess * 0.5 != 0):
        guess = guess * 0.5
        i += 1
    return guess, i

def maxfloat(guess = 1.0):
    guess = float(guess)
    i = 0
    while(guess * 2 != guess):
        guess = guess * 2
        i += 1
    return guess, i


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
            raise ValueError("Length is infinite.")
        elif self.rate < 0:
            # Exponential decay approaching zero
            length = int(math.ceil(self.half_life*(minfloat(self.amp)[1]+1)))
        else:
            # Exponential growth approaching inf
            # length = int(math.floor(
            #     abs(self.half_life) * (maxfloat(self.amp)[1]) + 1
            # ))
            raise ValueError("Exponential functions with infinite values \
                cause overflows too easily.")
        return length

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