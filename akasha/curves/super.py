#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Superformula curve
"""

from __future__ import division

import numpy as np

from akasha.curves.circle import Circle
from akasha.curves.curve import Curve
from akasha.utils.array import is_sequence
from akasha.utils.python import class_name
from akasha.math import pi2, normalize


class Super(Curve):
    """Oscillator curve that has superness parameters."""

    def __init__(self, m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Super oscillator can be initialized using superness parameters to control the shape.

        See 'Superellipse' article at Wikipedia for explanation of what these parameters mean:
        http://en.wikipedia.org/wiki/Superellipse

        Also see Gielis curve:
        http://www.2dcurves.com/power/powergc.html
        http://en.wikipedia.org/wiki/Superformula
        """
        self.superness = self.get_superness(m, n, p, q, a, b)

    @staticmethod
    def get_superness(m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Return a superness out of the arguments m to b.
        If given a sequence as m, will spread to other arguments, ignoring them.

        Defaults to (m=4.0, n=2.0, p=2.0, q=2.0, a=1.0, b=1.0) as an np.array.
        If b is missing, but a provided, it will be given the value of a.
        Arguments n, p and q are handled similarly. Missing values are filled from the left.
        """
        if is_sequence(m):
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

    def at(self, points):
        """Superformula curve at points."""
        return normalize(self.formula(points, self.superness)) * Circle.at(points)

    @staticmethod
    def formula(at, superness):
        """
        Superformula function.
        Generates amplitude curves applicable to oscillators by multiplying.

        Usage:
        supercurve(angles, superness)
        s = Super(431, m, n, p, q, a, b), where
        m is the number of spikes and n-q determine the roundness.

        For more information, see:
        http://en.wikipedia.org/wiki/Superellipse and
        http://en.wikipedia.org/wiki/Superformula
        """
        (m, n, p, q, a, b) = list(superness)
        assert np.isscalar(m), f'{m!s} in superformula is not scalar.'
        coeff = pi2 * at * (m / 4.0)
        return (np.abs(np.cos(coeff) / a) ** p + np.abs(np.sin(coeff) / b) ** q) ** (-1.0 / n)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(self.superness))

    def __repr__(self):
        return f'{class_name(self)}{tuple(self.superness)}'
