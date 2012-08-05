#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np

from numbers import Number

from .generators import PeriodicGenerator

from ..utils.math import pi2, normalize


class Curve(object, PeriodicGenerator):
    """Generic curve abstraction"""

    @staticmethod
    def at(param):
        return param


class Circle(Curve):
    """Curve of the circle"""

    @staticmethod
    def at(param):
        return np.exp(1j * pi2 * param)

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


