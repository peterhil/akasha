#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Experiments on fast ways to calculate sine wave oscillators
"""

import itertools as it
import numpy as np

from akasha.audio.oscillator import Osc


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return np.fromiter(it.islice(iterable, n), np.complex128)


def recycle(n, iterable):
    return take(n, it.cycle(iterable))


def combine_oscs(o1, o2):
    """
    Combine two oscillators by their derivation (differences between samples).
    """
    o1_length = o1.period if isinstance(o1, Osc) else len(o1)
    o2_length = o2.period if isinstance(o2, Osc) else len(o2)

    period = np.lcm([o1_length, o2_length])

    d1 = np.ediff1d(o1[::])
    d2 = np.ediff1d(o2[::])

    return np.cumsum(recycle(period, d1) * recycle(period, d2))
