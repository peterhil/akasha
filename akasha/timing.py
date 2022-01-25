#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Sampler module.
"""

from __future__ import division

import numpy as np

from timeit import default_timer as timer

from akasha.utils import is_empty
from akasha.utils.log import logger


class Sampler(object):
    """
    A sampler object, providing parameters for sampling.
    """

    def __init__(self, rate=44100, frametime=40, antialias=False, allow_negative=False):
        """
        Parameters:
        -----------
        rate : int
            The sampling rate (frequency) of the sampler.
        frametime : int
            The time interval in milliseconds between video frames.
            The default is 40 ms which corresponds to 25 Hz (1000/40).
        antialias : bool
            Whether to force frequencies below Nyquist frequency, which is sampling rate / 2.
        allow_negative : bool
            Whether to allow negative frequencies to occur.
        """
        self.rate = rate
        self.frametime = frametime
        self.prevent_aliasing = antialias
        self.negative_frequencies = allow_negative
        self.paused = False

    @property
    def videorate(self):
        """
        Sampler video frame rate.
        """
        return 1000 / self.frametime

    @property
    def nyquist(self):
        return self.rate / 2

    def change_frametime(self, ms=None, rel=0, mintime=16):
        """
        Changes video frame time (in ms).
        """
        if ms is None:
            ms = self.frametime
        ms = max(int(round(ms + rel)), mintime)  # Limit to mintime (1000 / 16 = 62.5 Hz)
        logger.info("Changing video FRAME TIME to {0} ms ({1:.3f} FPS)".format(ms, 1000 / ms))
        self.frametime = ms
        return ms

    def blocksize(self):
        """
        Calculate how may audio samples fit on one video frame.
        """
        return int(round(self.rate / self.videorate))

    def at(self, t, dtype=np.float64):
        """
        Return frame numbers from times (t).
        """
        return (np.array(t, dtype=np.float64) * self.rate).astype(dtype)

    def slice(self, start, end=None, step=1):
        """
        Return times from slice of frame numbers (which can be floats).
        """
        if end is None:
            end = start; start = 0
        return self.times(start, end, step) / self.rate

    def times(self, start, end=None, step=None):
        """
        Return an array of sample times from time slice parameters.
        """
        if end is None:
            end = start; start = 0
        if step is None:
            step = 1.0 / self.rate
        return np.arange(start, end, step, dtype=np.float64)

    def pause(self):
        """
        Pause the playback.
        """
        self.paused = not self.paused
        logger.info("Pause" if self.paused else "Play")

sampler = Sampler()


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
