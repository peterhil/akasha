#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import sys

from numbers import Number

from akasha.audio.generators import PeriodicGenerator
from akasha.utils import issequence
from akasha.utils.log import logger
from akasha.utils.math import clip, pad, pi2, normalize
from akasha.utils.patterns import Singleton


class Curve(PeriodicGenerator):
    """Generic curve abstraction"""

    @staticmethod
    def at(param):
        raise NotImplementedError("Please implement static method at() in a subclass.")

    def __call__(self, param):
        return self.at(param)

    def __repr__(self):
        return "%s()" % (self.__class__.__name__ ,)

    def __str__(self):
        return repr(self)


class Circle(Curve, Singleton):
    """Curve of the circle"""

    @staticmethod
    def at(param):
        return np.exp(1j * pi2 * param)


class Square(Curve, Singleton):
    """Curve of the square wave (made with np.sign)"""

    @staticmethod
    def at(param):
        return clip(Circle.at(param) * 2)


class Super(Curve):
    """Oscillator curve that has superness parameters."""

    def __init__(self, m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Super oscillator can be initialized using superness parameters to control the shape.

        See 'Superellipse' article at Wikipedia for explanation of what these parameters mean:
        http://en.wikipedia.org/wiki/Superellipse
        """
        self.superness = self.get_superness(m, n, p, q, a, b)

    @staticmethod
    def get_superness(m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Return a superness out of the arguments m to b.
        If given a sequence as m, will spread it through other arguments, ignoring them.

        Defaults to (m=4.0, n=2.0, p=2.0, q=2.0, a=1.0, b=1.0) as an np.array.
        If b is missing, but a provided, it will be given the value of a.
        Arguments n, p and q are handled similarly, so the missing values are filled from the left.
        """
        if issequence(m):
            superness = np.repeat(None, 6)
            superness[:min(len(m), 6)] = np.array(m[:6], dtype=np.float64)
            # if len(superness) < 6:
            #     if len(superness) < 4:
            #         superness = pad(superness, count=(4 - len(superness)))
            #     superness = pad(superness, count=(6 - len(superness)), value=1)
            # return superness
            (m, n, p, q, a, b) = superness

        return np.array([
            m or 4.0,
            n or 2.0,
            p or n or 2.0,
            q or p or n or 2.0,
            a or 1.0,
            b or a or 1.0,
        ], dtype=np.float64)

    def at(self, param):
        return normalize(self.formula(param, self.superness)) * Circle.at(param)

    @staticmethod
    def formula(at, superness):
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

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(self.superness))

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, tuple(self.superness))


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

