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


def polygon_osc(n=6, harmonics=1, damping=None, rand_phase=False):
    s = Super(n, 1.5, 1.5, 1.5)
    o = Osc(sampler.rate / n, curve=s)
    h = Harmonics(o, n=harmonics, damping=damping, rand_phase=rand_phase)

    return h


def test_clothoids(snd, n=6, simple=False):
    # deg = pi2
    quarter = pi2 / 4
    indices = np.arange(-1, n + 2)
    points = snd[indices]

    if simple:
        # Works for circles and super ellipses
        # tangents = (deg * (np.arange(-1, n + 1) / n + 0.25)) % deg
        tangents = np.angle(points) + quarter
    else:
        mids = np.array([
            np.angle(points[i] - midpoint(points[i - 1], points[i+1]))
            for i in np.arange(0, n + 1)
        ])
        tangents = (mids + quarter) % pi2

    logger.debug("points %r:\n%r", points.shape, points)
    logger.debug("tangents %r:\n%r", tangents.shape, tangents / pi2)

    clothoid_list = np.array([
        # Also see SolveG2 on Pyclothoids documentation for G2 continuity:
        # https://pyclothoids.readthedocs.io/en/latest/basic.html
        Clothoid.G1Hermite(
            points[i].real,
            points[i].imag,
            tangents[i],
            points[i + 1].real,
            points[i + 1].imag,
            tangents[i + 1],
        )
        for i in np.arange(1, n)
    ])
    logger.debug("clothoid list shape: %r", clothoid_list.shape)
    if n < 10:
        logger.debug("clothoid list:\n%r", clothoid_list)

    return clothoid_list


def resample_with_clothoids(
    clothoids,
    ratio,
):
    n = len(clothoids)

    indices = np.array(np.divmod(
        np.divide(np.arange(ratio * n, dtype=np.float64), ratio),
        1,
    )).T

    logger.debug("Resample with indices %r", indices.shape)

    samples = np.array([
        clothoids[int(i)].X(t) + 1j * clothoids[int(i)].Y(t)
        for [i, t] in indices
    ], dtype=np.complex128)

    return samples


def plot_clothoids_test(
    n=6,
    simple=False,
    debug=False,
    use_env=False,
    signal=None,
    plot=True,
    samples=50,
    **kwargs,
):
    if signal is not None:
        snd = signal
    else:
        snd = polygon_osc(n, **kwargs)
        if use_env:
            env = Exponential(-0.987, amp=0.9)
            snd = Mix(snd, env)

    if debug:
        logger.setLevel(logging.DEBUG)
        np.set_printoptions(precision=2)
        plot_signal(snd[:n + 1])

    clothoids = test_clothoids(snd, n + 1, simple)

    if plot:
        for i in clothoids:
            plt.plot(*i.SampleXY(samples))

    return clothoids
