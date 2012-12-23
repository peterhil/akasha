#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from akasha.audio.generators import Generator
from akasha.timing import sampler
from akasha.utils.math import minfloat


class Exponential(Generator):
    """
    Exponential decay and growth for envelopes.
    """

    def __init__(self, rate=0.0, amp=1.0):
        super(self.__class__, self).__init__()

        if isinstance(rate, tuple):
            self.rate, self.amp = rate
        else:
            self.rate = rate
            self.amp = amp

    @property
    def half_life(self):
        """Returns the time required to reach half-life from a starting amplitude."""
        return np.inf if self.rate == 0 else np.log(2.0) / -self.rate * sampler.rate

    @classmethod
    def from_half_life(cls, time, amp=1.0):
        """
        Returns an exponential decay envelope. Time parameter for half-life is measured in seconds.
        """
        if np.inf == np.abs(time):
            return cls(rate=0.0, amp=amp)
        else:
            return cls(rate=sampler.rate / (-(time * sampler.rate) / np.log(2.0)), amp=amp)

    @property
    def scale(self):
        """
        Returns the time required to reach a (discrete) zero from a starting amplitude.
        """
        return self.half_life * (minfloat(np.max(self.amp))[1] + 1)

    @classmethod
    def from_scale(cls, time, amp=1.0):
        """
        Create an exponential that reaches zero in the time given.
        """
        return cls.from_half_life((time / sampler.rate) / (minfloat(np.max(amp))[1] + 1), amp)

    def sample(self, frames):
        """
        Sample the exponential.
        """
        # Convert frame numbers to time (ie. 44100 => 1.0)
        time = np.array(frames) / float(sampler.rate)
        return self.amp * np.exp(self.rate * time)

    # def __len__(self):
    #     return int(np.ceil(np.abs(self.scale)))

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.rate, self.amp)

    def __str__(self):
        return "<%s: rate=%s, amp=%s>" % (self.__class__.__name__, self.rate, self.amp)


class Attack(Exponential):
    """
    Exponential attack (reversed decay/growth) envelope
    """

    def __init__(self, rate=0.0, amp=1.0, *args, **kwargs):
        super(self.__class__, self).__init__(rate, amp, *args, **kwargs)

    def sample(self, iterable, threshold=1.0e-6):
        """
        Sample the attack envelope.
        """
        attack = super(self.__class__, self).sample(iterable)
        attack = attack[attack > threshold][::-1]  # Filter silence and reverse

        frames = np.zeros(len(iterable))
        frames.fill(attack[-1])  # Sustain level
        frames[:len(attack)] = attack

        del(attack)
        return frames


class Gamma(Generator):
    """
    Gamma cumulative distribution function derived envelope.
    """

    def __init__(self, shape=1.0, scale=1.0):
        super(self.__class__, self).__init__()

        if isinstance(shape, tuple):
            self.shape, self.scale = shape
        else:
            self.shape = shape
            self.scale = scale  # Inverse rate

    def sample(self, iterable):
        """
        Sample the gamma exponential.
        """
        rate = (1.0 / max(self.scale, 1e-06))
        frames = (np.array(iterable) / float(sampler.rate)) * rate
        return sp.special.gammaincc(self.shape, frames)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.shape, self.scale)

    def __str__(self):
        return "<%s: shape=%s, scale=%s>" % (self.__class__.__name__, self.shape, self.scale)


class Timbre(Generator):
    """
    Defines an envelope timbre for frequencies.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

