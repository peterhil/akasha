#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Functional timing module.
"""

import numpy as np


class sampler:
    """
    Sampler.
    """

    rate = 44100


def samples(start, end, rate=sampler.rate):
    """
    Sample frame indices from start to end at rate.
    """
    start = int(np.ceil(start * rate))
    stop = int(round(np.floor(end * rate)))

    return np.arange(start, stop)


def times(start, end, rate=sampler.rate):
    """
    Sample times from start to end at rate.
    """
    return samples(start, end, rate) / float(rate)
