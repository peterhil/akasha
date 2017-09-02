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
import scipy.special as sc

from akasha.audio.envelope.adsr import Adsr
from akasha.audio.envelope.beta import Beta, InverseBeta
from akasha.audio.envelope.exponential import Exponential
from akasha.audio.generators import Generator
from akasha.timing import sampler


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
        return sc.gammaincc(self.shape, frames)

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
