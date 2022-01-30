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

from akasha.settings import config
from akasha.utils.log import logger


class Sampler():
    """
    A sampler object, providing parameters for sampling.
    """

    def __init__(
        self,
        rate=config.sampling.RATE,
        frametime=config.sampling.FRAMETIME,
        antialias=config.sampling.ANTIALIAS,
        allow_negative=config.sampling.NEGATIVE,
        ):
        """
        Parameters:
        -----------
        rate : int
            The sampling rate (frequency) of the sampler.
        frametime : int
            The time interval in milliseconds between video frames.
            The default is 40 ms which corresponds to 25 Hz (1000/40).
        antialias : bool
            Whether to force frequencies below Nyquist frequency, which
            is sampling rate / 2.
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
        # Limit to mintime (1000 / 16 = 62.5 Hz)
        ms = max(int(round(ms + rel)), mintime)
        logger.info(
            "Changing video FRAME TIME to %d ms (%1:.3f FPS)",
            ms, 1000 / ms
        )
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
