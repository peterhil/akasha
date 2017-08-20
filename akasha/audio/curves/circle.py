#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Circle curve
"""

from __future__ import division

import numpy as np

from akasha.audio.curves.curve import Curve
from akasha.utils.math import pi2
from akasha.utils.patterns import Singleton


class Circle(Curve, Singleton):
    """Curve of the circle"""

    @staticmethod
    def at(points):
        """Circle points of the unit circle at the complex plane."""
        return np.exp(1j * pi2 * points)
