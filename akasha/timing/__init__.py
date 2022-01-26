#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Timings module.
"""

from __future__ import division

import numpy as np

from timeit import default_timer as timer

from akasha.utils.array import is_empty
from akasha.utils.log import logger
from akasha.timing.sampler import sampler


def time_slice(dur, start=0, time=False):
    """Use a time slice argument or the provided attributes 'dur' and 'start' to
    construct a time slice object."""
    start *= sampler.rate
    time = time or slice(int(round(0 + start)), int(round(dur * sampler.rate + start)))
    if not isinstance(time, slice):
        raise TypeError("Expected a %s for 'time' argument, got %s." % (slice, type(time)))
    return time


class Timed(object):
    """
    Time some code using with statement.
    """
    elapsed = 0

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.end = timer()
        self.elapsed = self.end - self.start

    def __float__(self):
        return float(self.elapsed)


class Watch(object):
    def __init__(self, maxstops=5):
        self.paused = 0
        self.maxstops = maxstops
        self.timings = []
        self.reset()

    def reset(self):
        self.epoch = timer()
        if self.paused:
            self.paused = self.epoch
        self.lasttime = 0

    def time(self):
        if not self.paused:
            return timer() - self.epoch
        else:
            return self.paused - self.epoch

    def last(self):
        return self.time() - self.lasttime

    def next(self):
        if not self.paused:
            self.lasttime = self.time()
            self.timings.append(self.lasttime)
            self.timings = self.timings[-self.maxstops:]
        return self.lasttime

    def pause(self):
        if not self.paused:
            self.paused = timer()
        else:
            if self.paused > 0:
                self.epoch += timer() - self.paused
            self.paused = 0

    def get_fps(self, n=None):
        if n is None:
            ts = np.ediff1d(np.array(self.timings))
        elif n > 0:
            ts = np.ediff1d(np.array(self.timings[-n:]))

        # print('timings:', ts)
        ts = ts[ts >= 2e-3]  # Filter trash values after reset
        if is_empty(ts): return 0

        return np.median(1.0 / ts)
