#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np

from fractions import Fraction

from .curves import *
from .frequency import Frequency, FrequencyRatioMixin
from .generators import PeriodicGenerator

from ..timing import sampler
from ..utils.math import pi2, normalize


class Osc(object, FrequencyRatioMixin, PeriodicGenerator):
    """Generic oscillator class with a frequency and a parametric curve."""

    def __init__(self, freq, curve = Circle()):
        self._hz = Frequency(freq)
        self.curve = curve

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * sampler.rate)

    @property
    def sample(self):
        return self.curve.at(Frequency.angles(self.ratio))

    @property
    def imag(self):
        return self.sample.imag

    def __repr__(self):
        return "%s(%s, curve=%s)" % (self.__class__.__name__, self.frequency._hz, repr(self.curve))

    def __str__(self):
        return "<%s: %s, curve = %s>" % (self.__class__.__name__, self.frequency, str(self.curve))


def chirp_zeta(z1 = -0.5-100j, z2 = 0.5+100j, dur = 10):
    """
    Chirp sound made by sampling a line z (z1 -> z2) from the complex plane,
    and using the function (k ** -z, k = 0..n) used for summation in the Riemann Zeta function.

    Other interesting values to try:
    chirp_zeta(-10.5-1000j, 1.5-10000j)

    Reference: http://en.wikipedia.org/wiki/Riemann_zeta_function
    """
    n = int(round(dur * sampler.rate))
    z = np.linspace(z1, z2, n)
    k = np.arange(n)
    return normalize(k ** -z)

