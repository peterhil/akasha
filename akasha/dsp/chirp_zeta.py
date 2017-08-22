#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Chirp Zeta function
"""

from __future__ import division

import numpy as np

from akasha.timing import sampler
from akasha.math import normalize


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
