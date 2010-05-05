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


class Exponential(Generator):
    """Exponential decay and growth for envelopes."""

    def __init__(self, rate, amp=1.0):
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
        return "Exponential(%s, %s)" % (self.rate, self.amp)

    def __str__(self):
        return "<Exponential: rate=%s, amp=%s>" % (self.rate, self.amp)