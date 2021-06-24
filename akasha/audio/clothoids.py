#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio compression algorithm based on clothoids
"""

from __future__ import division

import logging
import matplotlib.pyplot as plt
import numpy as np

from pyclothoids import Clothoid

from akasha.audio import Exponential, Harmonics, Mix, Osc
from akasha.curves import Super
from akasha.graphic.plotting import plot_signal
from akasha.math import pi2
from akasha.math.geometry import midpoint
from akasha.timing import sampler
from akasha.utils.log import logger


def polygon_osc(n=6, curve=Super(6, 1.5, 1.5, 1.5), harmonics=1, rand_phase=False):
    o = Osc(sampler.rate / n, curve=curve)
    h = Harmonics(o, n=harmonics, rand_phase=rand_phase)

    return h


def test_clothoids(snd, n=6, simple=False):
    deg = pi2
    quarter = pi2 / 4
    indices = np.arange(-1, n + 1)
    points = snd[indices]

    if simple:
        # Works for circles and super ellipses
        tangents = (deg * (np.arange(-1, n + 1) / n + 0.25)) % deg
    else:
        mids = np.array([np.angle(points[i] - midpoint(points[i - 1], points[i + 1])) for i in np.arange(n)])
        tangents = (mids + quarter) % pi2

    logger.debug("points %r:\n%r", points.shape, points)
    logger.debug("tangents %r:\n%r", tangents.shape, tangents / pi2)

    clothoid_list = [
        Clothoid.G1Hermite(
            points[i].real,
            points[i].imag,
            tangents[i],
            points[i + 1].real,
            points[i + 1].imag,
            tangents[i + 1],
        )
        for i in np.arange(n - 1)
    ]

    return clothoid_list


def plot_clothoids_test(n=6, simple=False, debug=False, use_env=False, **kwargs):
    snd = polygon_osc(n, **kwargs)
    if use_env:
        env = Exponential(-0.987, amp=0.9)
        snd = Mix(osc, env)

    if debug:
        logger.setLevel(logging.DEBUG)
        np.set_printoptions(precision=2)
        plot_signal(snd[:n + 1])

    clothoid_list = test_clothoids(snd, n, simple)
    for i in clothoid_list:
        plt.plot( *i.SampleXY(500) )
