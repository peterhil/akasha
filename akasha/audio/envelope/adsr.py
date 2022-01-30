#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Adsr envelopes
"""

import numpy as np

from akasha.audio.delay import Delay
from akasha.audio.envelope.beta import Beta, InverseBeta
from akasha.audio.envelope.exponential import Exponential
from akasha.audio.generators import Generator
from akasha.audio.mix import Mix
from akasha.audio.scalar import Scalar
from akasha.audio.sum import Sum


def iter_param(param):
    return iter(np.asarray([param], dtype=np.float).flatten())


class Adsr(Generator):
    """
    Adsr envelopes with beta distrubution cdf curves.
    """

    def __init__(
        self,
        attack=(0.15,),
        decay=(0.25,),
        sustain=0.5,
        release=(0.2,),
        released_at=None,
        decay_overlap=0,
    ):
        self.released_at = released_at
        self.sustain = Scalar(float(sustain), dtype=np.float64)
        self.decay_overlap = decay_overlap
        # Envelope parts
        self.attack = Beta(*iter_param(attack))
        self.decay_params = decay
        self.release = InverseBeta(*iter_param(release))

    @property
    def decay(self):
        return Delay(
            self.attack.time - self.decay_overlap,
            InverseBeta(
                *iter_param(self.decay_params), amp=1.0 - self.sustain_level
            ),
        )

    @property
    def sustain_level(self):
        return self.sustain.value

    def release_at(self, time=None):
        """
        Set release time.
        """
        if np.isreal(time):
            if time is not None:
                self.released_at = float(time)
            else:
                self.released_at = time
        else:
            raise ValueError("Release time should be a real number!")

    def at(self, t):
        """
        Sample adsr envelope at times.
        """
        adsr = Mix(self.attack, self.decay)
        if self.released_at is not None:
            adsr = Mix(adsr, Delay(float(self.released_at), self.release))
        return adsr.at(t)
