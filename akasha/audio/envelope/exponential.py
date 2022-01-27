#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Exponential envelopes
"""

import numpy as np

from akasha.audio.generators import Generator
from akasha.math import minfloat
from akasha.timing import sampler
from akasha.utils.python import class_name


class Exponential(Generator):
    """
    Exponential decay and growth for envelopes.
    """

    def __init__(self, rate=0.0, amp=1.0):
        if isinstance(rate, tuple):
            self.rate, self.amp = np.array(rate).astype(np.float64)
        else:
            self.rate = float(rate)
            self.amp = float(amp)

    @property
    def half_life(self):
        """Returns the time required to reach half-life from a
        starting amplitude."""
        return np.inf if self.rate == 0 else np.log(2.0) / -self.rate

    @classmethod
    def from_half_life(cls, time, amp=1.0):
        """
        Returns an exponential decay envelope.
        Time parameter for half-life is measured in seconds.
        """
        if np.inf == np.abs(time):
            return cls(rate=0.0, amp=amp)
        else:
            return cls(rate=np.log(2.0) / -time, amp=amp)

    @property
    def scale(self):
        """Returns the time required to reach a (discrete) zero
        from a starting amplitude.
        """
        return self.half_life * (minfloat(np.max(self.amp))[1] + 1)

    @classmethod
    def from_scale(cls, time, amp=1.0):
        """
        Create an exponential that reaches zero in the time given.
        """
        return cls.from_half_life(time / (minfloat(np.max(amp))[1] + 1), amp)

    def sample(self, frames):
        """
        Sample the exponential at sampler frames.
        Converts frame numbers to time (ie. 44100 => 1.0).
        """
        if np.isscalar(frames):
            frames = np.array(frames, dtype=np.float64)
        else:
            frames = np.fromiter(frames, dtype=np.float64)
        times = frames / float(sampler.rate)

        return self.at(times)

    def at(self, times):
        """
        Sample the exponential at sample times.
        """
        return np.clip(
            self.amp * np.exp(self.rate * times),
            a_min=0.0,
            a_max=1.0,
        )

    # def __len__(self):
    #     return int(np.ceil(np.abs(self.scale)))

    def __repr__(self):
        return f'{class_name(self)}({self.rate!r}, {self.amp!r})'

    def __str__(self):
        return f'<{class_name(self)}: rate={self.rate!s}, amp={self.amp!s}>'
