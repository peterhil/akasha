#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functional timing module.
"""

import numpy as np


class sampler(object):
    """
    Sampler.
    """
    rate = 44100


def samples(start, end, rate=sampler.rate):
    """
    Sample frame indices from start to end at rate.
    """
    return np.arange(int(np.ceil(start * rate)), int(round(np.floor(end * rate))))


def times(start, end, rate=sampler.rate):
    """
    Sample times from start to end at rate.
    """
    return samples(start, end, rate) / float(rate)
