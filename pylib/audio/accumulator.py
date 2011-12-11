#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np
from cmath import rect, pi, exp, phase
from fractions import Fraction

# Types
import types
from numbers import Number

# My modules
from audio.generators import PeriodicGenerator
from io.audio import play, write
from timing import Sampler
from utils.decorators import memoized
from utils.math import *
from utils.graphing import *

# Settings
np.set_printoptions(precision=16, suppress=True)


class Acc(object, PeriodicGenerator):
    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self._ratio = Osc.limit_ratio(Fraction(*args))
        self.roots = self.gen
        self.roots(self._ratio)

    def gen(self, ratio):
        if ratio == 0:
            return np.exp(np.array([0j]))
        pi2 = 2 * np.pi
        return ratio.numerator * 1j * pi2 * np.arange(0, 1, 1.0/ratio.denominator)

    @classmethod
    def freq(cls, freq):
        ratio = Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(Sampler.rate)
        return cls(ratio)

    ### Properties ###

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator

    @property
    def frequency(self):
        return float(self.ratio * Sampler.rate)

    ### Sampling ###

    @property
    def sample(self):
        return self.gen(self.ratio)


if __name__ == '__main__':
    o = Acc(Fraction(1, 8))
    t = slice(0, Sampler.rate)
    print o.np_exp(o.period)
    print to_phasors(o.sample)
