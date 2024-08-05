#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Gamma envelopes
"""

import numpy as np
import scipy.special as sc

from akasha.audio.generators import Generator
from akasha.timing import sampler
from akasha.utils.python import class_name, _super


class Gamma(Generator):
    """
    Gamma cumulative distribution function derived envelope.
    """

    def __init__(self, shape=1.0, scale=1.0):
        _super(self).__init__()

        if isinstance(shape, tuple):
            self.shape, self.scale = shape
        else:
            self.shape = shape
            self.scale = scale  # Inverse rate

    @property
    def rate(self):
        """
        Rate, inverse of scale.
        """
        return 1.0 / max(self.scale, 1e-06)  # prevent ZeroDivisionError

    def at(self, t):
        """
        Sample the gamma exponential at times (t).
        """
        return sc.gammaincc(self.shape, t * self.rate)

    def sample(self, frames):
        """
        Sample the gamma exponential.
        """
        times = (np.array(frames) / float(sampler.rate)) * self.rate
        return sc.gammaincc(self.shape, times)

    def __repr__(self):
        return f'{class_name(self)}({self.shape!r}, {self.scale!r})'

    def __str__(self):
        return (
            f'<{class_name(self)}: shape={self.shape!s}, '
            + f'scale={self.scale!s}>'
        )
