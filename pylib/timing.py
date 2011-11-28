#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from timeit import default_timer as clock
import math

# from utils import Sampler
# from generators import Generator

np.set_printoptions(precision=9)


class Sampler(object):

    # Settings
    prevent_aliasing = True
    negative_frequencies = False

    rate = 44100
    videorate = 25

def timecode(t, precision=1000000):
    """t = time.clock(); t; int(math.floor(t)); int(round((t % 1.0) * 1000000))"""
    return (int(math.floor(t)), int(round((t % 1.0) * precision)))

def times_at(frames):
    """Convert frame numbers to time.

    >>> time_at(44100)
    1.0
    """
    return frames / float(Sampler.rate)

def frames_at(times):
    """Convert time to frame numbers (ie. 1.0 => 44100)"""
    return int(round(times * Sampler.rate))


def time_slice(dur, start, time=False):
    """Use a time slice argument or the provided attributes 'dur' and 'start' to
    construct a time slice object."""
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))
    if not isinstance(time, slice):
        raise TypeError("Expected a %s for 'time' argument, got %s." % (slice, type(time)))
    return time

class Timeslice(object):
    def __init__(self, start=0, stop=None):
        if not stop:
            start, stop = (0, start)
        self.start = start
        self.stop = stop

    @property
    def sample(self):
        #return np.array(np.linspace(self.start, self.stop, Sampler.rate, endpoint=False) * Sampler.rate, dtype=int)
        return np.array(np.arange(self.start * Sampler.rate, self.stop * Sampler.rate), dtype=int)

    def __repr__(self):
        return "<%s: start = %s, stop = %s>" % (self.__class__.__name__, self.start, self.stop)


class OutputStream(object):
    pass
