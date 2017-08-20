#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curves for oscillators
"""

from __future__ import division

import numpy as np

from akasha.audio.curves.circle import Circle
from akasha.audio.curves.curve import Curve
from akasha.audio.curves.ellipse import Ellipse
from akasha.audio.curves.super import Super

from akasha.timing import sampler
from akasha.utils.math import clip, normalize
from akasha.utils.patterns import Singleton


class Square(Curve, Singleton):
    """Curve of the square wave. Made with np.sign()."""

    @staticmethod
    def at(points):
        return clip(Circle.at(points) * 2)


def chirp_zeta(z1=-0.5 - 100j, z2=0.5 + 100j, dur=10):
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
