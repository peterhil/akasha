#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import exceptions
import math
import numpy as np

from timeit import default_timer as clock

from akasha.utils.log import logger


class Sampler(object):
    """
    A sampler object, providing parameters for sampling.
    """

    def __init__(self, rate = 44100, frametime = 40, antialias = True, allow_negative = False):
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
        return 1000 / self.frametime

    def change_frametime(self, ms = None, rel = 0, mintime = 16):
        if ms == None:
            ms = self.frametime
        ms = max(int(round(ms + rel)), mintime) # Limit to mintime (1000 / 16 = 62.5 Hz)
        logger.info("Changing video FRAME TIME to {0} ms ({1:.3f} FPS)".format(ms, 1000 / ms))
        self.frametime = ms
        return ms

    def blocksize(self):
        return int(round(self.rate / self.videorate))

    def pause(self):
        self.paused = not self.paused
        logger.info("Pause" if self.paused else "Play")

sampler = Sampler()


def indices(snd, dur=False):
    if hasattr(snd, "size"):
        size = snd.size
    elif dur:
        size = int(round(dur * sampler.rate))
    else:
        raise exceptions.ValueError("indices(): Sound must have size, or duration argument must be specified.")
    return np.append(np.arange(0, size, sampler.blocksize()), size)

def timecode(t, precision=1000000):
    """t = time.clock(); t; int(math.floor(t)); int(round((t % 1.0) * 1000000))"""
    return (int(math.floor(t)), int(round((t % 1.0) * precision)))

def times_at(frames):
    """Convert frame numbers to time.

    >>> time_at(44100)
    1.0
    """
    return frames / float(sampler.rate)

def frames_at(times):
    """Convert time to frame numbers (ie. 1.0 => 44100)"""
    return np.array(int(round(times * sampler.rate)))

def sample_times(start, end, rate=sampler.rate):
    """
    Return times when samples occur at rate.

    >>> np.set_printoptions(precision=8, suppress=True)
    >>> stime(0.5,1.5)
    array([ 0.5       ,  0.50002268,  0.50004535,  0.50006803,  0.5000907 ,
            0.50011338,  0.50013605,  0.50015873,  0.50018141,  0.50020408, ...,
            1.49977324,  1.49979592,  1.49981859,  1.49984127,  1.49986395,
            1.49988662,  1.4999093 ,  1.49993197,  1.49995465,  1.49997732])
    """
    return np.linspace(start, end, (end-start)*rate, endpoint=False)


def time_slice(dur, start=0, time=False):
    """Use a time slice argument or the provided attributes 'dur' and 'start' to
    construct a time slice object."""
    start *= sampler.rate
    time = time or slice(int(round(0 + start)), int(round(dur * sampler.rate + start)))
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
        #return np.array(np.linspace(self.start, self.stop, sampler.rate, endpoint=False) * sampler.rate, dtype=int)
        return np.array(np.arange(self.start * sampler.rate, self.stop * sampler.rate), dtype=int)

    def __repr__(self):
        return "<%s: start = %s, stop = %s>" % (self.__class__.__name__, self.start, self.stop)


class OutputStream(object):
    pass

class Timeline(object):
    """Class representing time, both physical and discrete."""

    def __init__(self, resolution=sampler.rate):
        self.resolution = resolution

    def times(self, start_time, end_time=None):
        if end_time == None:
            end_time = start_time
            start_time = 0
        return np.arange(start_time, end_time, self.resolution, dtype=np.float64)

    def frames(self, start_time, end_time=None):
        if end_time == None:
            end_time = start_time
            start_time = 0
        return np.arange(start_time / self.resolution, end_time / self.resolution, 1, dtype=np.int64)

