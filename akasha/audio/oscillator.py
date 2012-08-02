#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import operator

from fractions import Fraction
from numbers import Number

from .frequency import Frequency, FrequencyRatioMixin
from .generators import PeriodicGenerator

from ..timing import sampler
from ..utils.decorators import memoized
from ..utils.log import logger
from ..utils.math import *


class Curve(object, PeriodicGenerator):
    """Generic curve abstraction"""

    def __init__(self):
        pass

    @staticmethod
    def at(param):
        return param


class Circle(Curve):
    """Curve of the circle"""

    @staticmethod
    def at(param):
        return np.exp(1j * pi2 * param)


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


class SuperEllipse(Curve):
    """Oscillator curve that has superness parameters."""
    
    def __init__(self, *superness):
        """
        Super oscillator can be initialized using superness parameters to control the shape.

        The parameters are:
        superness = { m: 4.0, n: 2.0, p: 2.0, q: 2.0, a: 1.0, b: 1.0 }
        
        See 'Superellipse' article at Wikipedia for explanation of what this parameter means:
        http://en.wikipedia.org/wiki/Superellipse
        """
        self.superness = self.normalise_superness(superness)

    @staticmethod
    def normalise_superness(superness):
        if not isinstance(superness, (dict, list, tuple, Number, None)):
            raise ValueError(
                "Superness %s needs to be a number, a tuple or a list of length one to six. " + \
                "Got type %s" % (superness, type(superness))
            )
        if isinstance(superness, tuple):
            if len(superness) == 6:
                return superness
            if isinstance(list(superness)[0], (tuple, list)):
                superness = list(superness)[0]
        if superness == None:
            logger.warn("Got None for superness!")
            superness = [2.0] # identity for superness
        if isinstance(superness, Number):
            superness = [superness]
        if isinstance(superness, dict):
            superness = superness.values()
        if len(superness) < 6:
            if len(superness) < 4:
                superness = list(superness) + [superness[-1]] * (4 - len(superness))
            superness = list(superness) + [1.0] * (6 - len(superness))
        return tuple(superness[:6])  # Take first six params

    def at(self, param):
        return normalize(self.superformula(param, self.superness)) * Circle.at(param)

    @staticmethod
    def superformula(at, superness):
        """
        Superformula function. Generates amplitude curves applicable to oscillators by multiplying.

        Usage:
        supercurve(angles, superness)
        s = Super(431, m, n, p, q, a, b), where m is number of spikes and n-q determine the roundness.

        For more information, see:
        http://en.wikipedia.org/wiki/Superellipse and
        http://en.wikipedia.org/wiki/Superformula
        """
        (m, n, p, q, a, b) = list(superness)
        assert np.isscalar(m), "%s in superformula is not scalar." % m
        coeff = pi2 * at * (m / 4.0)
        return (np.abs(np.cos(coeff) / a)**p + np.abs(np.sin(coeff) / b)**q) ** (-1.0/n)


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

