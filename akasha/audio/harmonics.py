#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Harmonics module
"""

import numpy as np

from akasha.audio.envelope import Exponential, Gamma
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import Generator
from akasha.audio.oscillator import Osc

from akasha.timing import sampler
from akasha.utils.decorators import memoized
from akasha.math import random_phasor, map_array, normalize, pi2


# TODO Eventually compose overtones of Mix objects and use Playable, and drop FrequencyRatioMixin
class Harmonics(FrequencyRatioMixin, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(
            self,
            sndobj=Osc(216.0),
            n=8,
            func=lambda x: 1 + x,
            damping=None,
            rand_phase=False):
        super(self.__class__, self).__init__()
        self.base = sndobj
        # TODO Setting ovt.frequency (ovt._hz) leaves ovt.base.frequency (ovt.base._hz)
        # where it was -- is this the desired behaviour?
        self._hz = self.base.frequency
        self.n = n
        self.func = func
        if damping == 'sine':
            # Sine waves FIXME: separate freq. damping from rate
            self.damping = lambda f, a=1.0: (
                -5 * np.log2(float(f)) / (10.0),
                a * float(self.frequency)/float(f)
            )
        elif damping == 'default':
            self.damping = lambda f, a=1.0: -5 * np.log2(float(f)) / 1000.0
        elif callable(damping):
            self.damping = damping
        else:
            self.damping = None
        self.rand_phase = rand_phase

    @property
    def max_harmonics(self):
        """Maximum number of harmonics to generate for a frequency."""
        low_freq_overtone_limit = 10
        return int(sampler.rate / (2.0 * max(self.frequency, low_freq_overtone_limit)))

    @property
    def limit(self):
        """Get the number of harmonics to generate for a frequency."""
        return max(min(self.max_harmonics, self.n), 1)

    @property
    def harmonics(self):
        """
        Generate harmonics using the function given in init.
        The number of harmonics is limited by self.limit.
        """
        return np.apply_along_axis(self.func, 0, np.arange(0, self.limit, dtype=np.float64))

    @property
    def oscs(self):
        """
        Oscillators based on harmonics.
        """
        base = self.base.__class__
        harmonics = np.array(float(self.frequency) * self.harmonics, dtype=np.float64)
        if 'Super' == self.base.curve.__class__.__name__:
            oscs = map_array(lambda f: base(f, curve=self.base.curve), harmonics, 'vec')
        else:
            oscs = map_array(base, harmonics, 'vec')
        return oscs[np.nonzero(oscs)]

    def at(self, t):
        """
        Sample Harmonics at times (t).
        """
        partials = []

        for o in self.oscs:
            out = o.at(t)

            if self.rand_phase:
                out *= np.array(random_phasor(1))  # TODO: Move phases to Osc/Frequency?

            if self.damping:
                e = Exponential(self.damping(o.frequency))
                # e = Gamma(-self.damping(o.frequency)[0], 1.0 / max(float(o.frequency) / 100.0, 1.0))

                # square waves
                # amplitude = float(self.frequency / o.frequency * float(self.frequency))
                # e = Exponential(0, amp=amplitude)

                # triangle waves
                # amplitude = float(self.frequency ** 2 / o.frequency ** 2 * float(self.frequency))
                # e = Exponential(0, amp=amplitude)

                # sine waves
                # e = Exponential(-o.frequency / 100.0)

                out *= e.at(t)

            partials.append(out)

        # Sum all partials and normalize volume
        return np.sum(partials, axis=0, dtype=np.complex128) / self.limit

    def __repr__(self):
        return "%s(sndobj=%s, n=%s, func=%s, damping=%s, rand_phase=%s>" % \
            (self.__class__.__name__, self.base, self.n, self.func, self.damping, self.rand_phase)

    def __str__(self):
        return "<%s: sndobj=%s, limit=%s, frequency=%s, harmonics=%s, func=%s, damping=%s>" % \
            (self.__class__.__name__, self.base, self.limit, self.frequency,
             self.harmonics, self.func, self.damping)
