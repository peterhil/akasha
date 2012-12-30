#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sampler module.
"""

from __future__ import absolute_import
from __future__ import division

import exceptions
import math
import numpy as np

from akasha.utils.log import logger


class Sampler(object):
    """
    A sampler object, providing parameters for sampling.
    """

    def __init__(self, rate=44100, frametime=40, antialias=True, allow_negative=False):
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
