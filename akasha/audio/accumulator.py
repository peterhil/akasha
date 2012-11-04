#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import types

from cmath import rect, pi, exp, phase
from fractions import Fraction
from numbers import Number

from .generators import PeriodicGenerator

from ..timing import sampler
from ..utils.math import pi2, to_phasors


class Acc(PeriodicGenerator):
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
        ratio = Fraction.from_float(float(freq)/sampler.rate).limit_denominator(sampler.rate)
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
        return float(self.ratio * sampler.rate)

    ### Sampling ###

    @property
    def sample(self):
        return self.gen(self.ratio)


if __name__ == '__main__':
    o = Acc(Fraction(1, 8))
    t = slice(0, sampler.rate)
    print o.np_exp(o.period)
    print to_phasors(o.sample)
