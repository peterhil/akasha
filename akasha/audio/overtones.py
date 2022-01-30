#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Overtones module
"""

import numpy as np

from builtins import zip

from akasha.audio.envelope import Exponential
from akasha.audio.frequency import FrequencyRatioMixin
from akasha.audio.generators import Generator
from akasha.audio.mix import Mix
from akasha.audio.oscillator import Osc
from akasha.audio.scalar import Scalar
from akasha.audio.sum import Sum
from akasha.math import random_phasor, map_array
from akasha.utils.python import class_name, _super


# TODO use Playable and drop FrequencyRatioMixin
class Overtones(FrequencyRatioMixin, Generator):
    """
    Overtones for a sound object having a frequency
    """

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
        elif damping == 'natural':
            self.damping = lambda f: (-5 * np.log2(float(f)) / 1000.0)
        elif callable(damping):
            self.damping = damping
        else:
            # Default is no damping
            self.damping = lambda f: 0
        self.rand_phase = rand_phase

    @property
    def overtones(self):
        """
        Generate overtones using the function given in init.
        """
        return map_array(self.func, np.arange(self.n, dtype=np.float64))

    @property
    def frequencies(self):
        """
        Generate frequencies based on overtones
        """
        frequencies = self.frequency * self.overtones
        return frequencies[np.nonzero(frequencies)]

    # TODO memoize with hash!
    @property
    def partials(self):
        """
        Mix generated overtone partials
        """
        frequencies = self.frequencies  # TODO memoize with dependencies

        # TODO Get the base curve another way in order to be able to
        # use Gamma curves on frequency plane for example, or any
        # other object with a frequency as the base.  Also consider
        # adding multiply to Playble, and pass methods through on Sum
        # and Mix etc.
        oscs = np.array([Osc(f, self.base.curve) for f in frequencies])

        envelopes = [Exponential(self.damping(f)) for f in frequencies]
        partials = [Mix(*part) for part in zip(oscs, envelopes)]

        # Random phases
        if self.rand_phase:
            phases = random_phasor(self.n)
            phases = [Scalar(phase, dtype=np.complex128) for phase in phases]
            partials = [Mix(*partial) for partial in zip(partials, phases)]

        return Mix(Sum(*partials), Scalar(1.0 / len(frequencies)))

    def at(self, t):
        """
        Sample Overtones at times (t).
        """
        return self.partials.at(t)

    def __repr__(self):
        return (
            f'{class_name(self)}(sndobj={self.base!r}, '
            + f'n={self.n!r}, '
            + f'func={self.func!r}, damping={self.damping!r}, '
            + f'rand_phase={self.rand_phase!r})'
        )

    def __str__(self):
        return (
            f'<{class_name(self)}: sndobj={self.base!r}, '
            + f'n={self.n!r}, '
            + f'frequency={self.frequency!r}, '
            + f'frequencies={self.frequencies!r}, '
            + f'func={self.func!r}, damping={self.damping!r}>'
        )
