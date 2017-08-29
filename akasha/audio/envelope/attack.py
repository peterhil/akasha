#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Envelopes
"""

import numpy as np

from akasha.audio.envelope.exponential import Exponential


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
