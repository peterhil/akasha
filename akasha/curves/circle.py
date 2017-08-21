#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Circle curve
"""

from __future__ import division

import numpy as np

from akasha.curves.curve import Curve
from akasha.utils.math import pi2
from akasha.utils.patterns import Singleton


class Circle(Curve, Singleton):
    """Curve of the circle"""

    @staticmethod
    def at(points):
        """
        Circle points of the unit circle at the complex plane.
        """
        return np.exp(1j * pi2 * points)

    @classmethod
    def roots_of_unity(cls, nth):
        """
        Nth roots of Unity.

        Return nth primitive roots of unity - the complex numbers
        located on the 1..N/nth tau angle on the unit circle.

        http://en.wikipedia.org/wiki/Roots_of_unity
        """
        points = np.linspace(0, 1, nth, endpoint=False)
        return cls.at(points)
