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
