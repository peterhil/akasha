#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Smoothing plugins
"""

from __future__ import division

import numpy as np
import scipy as sc

from akasha.funct.itertools import consecutive
from akasha.math import pad_ends, pi2
from akasha.math.geometry import angle_between, midpoint, triangle_incenter
from akasha.timing import sampler


quarter = pi2 / 4


def real_signal_with_timecode(signal):
    """
    Returns real signal with timecode from a complex signal.
    """
    y = signal.imag
    timecode = sampler.slice(len(signal))
    sample_points = timecode + y * 1j

    return sample_points


def apply_smoothing(signal, smoothing, *args, **kwargs):
    return smoothing(real_signal_with_timecode(signal), *args, **kwargs)


def average_smoothing(signal, angle_limit=quarter):
    """
    Smoothes a signal using linear average for signal values.
    """

    def smooth(a, b, c):
        if np.abs(angle_between(a, b, c)) < angle_limit:
            return midpoint(a, c)
        else:
            return b

    smoothed = [smooth(a, b, c) for a, b, c in consecutive(signal, 3)]
    return pad_ends(smoothed, signal[0], signal[-1])


def incenter_smoothing(signal, angle_limit=quarter):
    """
    Smoothes a signal using triangle incenter points for signal values.
    """

    def smooth(a, b, c):
        if np.abs(angle_between(a, b, c)) < angle_limit:
            incenter = triangle_incenter(a, b, c)
            return midpoint(a, c).real + incenter.imag * 1j
        else:
            return b

    smoothed = [smooth(a, b, c) for a, b, c in consecutive(signal, 3)]
    return pad_ends(smoothed, signal[0], signal[-1])


def exponential_smoothing(signal):
    pass


def straight_edge_smoothing(signal, angle_limit=quarter):
    """Smoothes a signal so that consecutive signal points do not
    form angles smaller than 90 degrees.
    """

    def smooth(a, b, c):
        # Calculate new position that makes at most 90 degree angle
        # FIXME Does not work as such!
        if np.abs(angle_between(a, b, c)) < angle_limit:
            return midpoint(a, c) + ((c - a) / 2) * 1j
        else:
            return b

    smoothed = [smooth(a, b, c) for a, b, c in consecutive(signal, 3)]
    return pad_ends(smoothed, signal[0], signal[-1])
