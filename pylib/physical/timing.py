#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from timeit import default_timer as clock

# Math &c.
import numpy as np


class Sampler(object):
    rate = 44100


def stime(start, end, rate=Sampler.rate):
    return np.linspace(start, end, (end-start)*rate, endpoint=False)

def times_at(*args):
    """Convert frame numbers to time.

    Examples:
    >>> Sampler.rate
    44100
    >>> time_at(44100)
    array([ 1.])
    >>> times_at(0,8)*44100
    array([ 0.,  8.])
    """
    return (np.array(args) / float(Sampler.rate)).flatten()

def frames_at(times):
    """Convert time to frame numbers (ie. 1.0 => 44100)"""
    time2frame = lambda t: int(round(t * Sampler.rate))

class Timeline(object):
    pass

class OutputStream(object):
    pass
