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
from akasha.utils.python import class_name, _super
from akasha.math import random_phasor, map_array, normalize, pi2


# TODO Eventually compose overtones of Mix objects and use Playable,
# and drop FrequencyRatioMixin
class Harmonics(FrequencyRatioMixin, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(
        self,
        sndobj=Osc(216.0),
        n=8,
        func=lambda x: 1 + x,
        damping=None,
        rand_phase=False,
    ):
        _super(self).__init__()
        self.base = sndobj
        # TODO Setting ovt.frequency (ovt._hz) leaves ovt.base.frequency
        # (ovt.base._hz) where it was -- is this the desired behaviour?
        self._hz = self.base.frequency
        self.n = n
        self.func = func
        if damping == 'sine':
            # Sine waves FIXME: separate freq. damping from rate
            self.damping = lambda f, a=1.0: (
                -5 * np.log2(float(f)) / (10.0),
                a * float(self.frequency) / float(f),
            )
        elif damping == 'default':
            self.damping = lambda f, a=1.0: -5 * np.log2(float(f)) / 1000.0
        elif callable(damping):
            self.damping = damping
        else:
            self.damping = None
        self.rand_phase = rand_phase

    @property
    def harmonics(self):
        """
        Generate harmonics using the function given in init.
        """
        return map_array(self.func, np.arange(self.n, dtype=np.float64))

    # TODO memoize with dependencies
    @property
    def frequencies(self):
        """
        Generate frequencies based on harmonics
        """
        frequencies = self.frequency * self.harmonics
        return frequencies[np.nonzero(frequencies)]

    def at(self, t):
        """
        Sample Harmonics at times (t).
        """
        partials = []
        oscs = np.array([Osc(f, self.base.curve) for f in self.frequencies])
        freq = self.frequency

        for o in oscs:
            out = o.at(t)

            # TODO: Move phases to Osc/Frequency?
            if self.rand_phase:
                out *= np.array(random_phasor(1))

            if self.damping:
                e = Exponential(self.damping(o.frequency))
                # scale = 1.0 / max(float(o.frequency) / 100.0, 1.0)
                # e = Gamma(-self.damping(o.frequency)[0], scale)

                # square waves
                # amplitude = float(freq / o.frequency * freq)
                # e = Exponential(0, amp=amplitude)

                # triangle waves
                # amplitude = float(freq ** 2 / o.frequency ** 2 * freq)
                # e = Exponential(0, amp=amplitude)

                # sine waves
                # e = Exponential(-o.frequency / 100.0)

                out *= e.at(t)

            partials.append(out)

        # Sum all partials and normalize volume
        return np.sum(partials, axis=0, dtype=np.complex128) / len(partials)

    def __repr__(self):
        return (
            f'{class_name(self)}(sndobj={self.base!r}, '
            + f'n={self.n!r}, '
            + f'func={self.func!r}, '
            + f'damping={self.damping!r}, '
            + f'rand_phase={self.rand_phase!r}>'
        )

    def __str__(self):
        return (
            f'{class_name(self)}: sndobj={self.base!s}, '
            + f'n={self.n!s}, '
            + f'frequency={self.frequency!s}, '
            + f'frequencies={self.frequencies!s}, '
            + f'func={self.func!s}, '
            + f'damping={self.damping!s}>'
        )
