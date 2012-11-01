#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np

from fractions import Fraction

from .curves import *
from .frequency import Frequency, FrequencyRatioMixin
from .generators import PeriodicGenerator

from ..timing import sampler
from ..utils.math import pi2, normalize


class Osc(FrequencyRatioMixin, PeriodicGenerator, object):
    """Generic oscillator class with a frequency and a parametric curve."""

    def __init__(self, freq, curve = Circle()):
        self._hz = Frequency(freq)
        self.curve = curve

    @property
    def sample(self):
        return self.curve.at(Frequency.angles(self.ratio))

    def __repr__(self):
        return "%s(%s, curve=%s)" % (self.__class__.__name__, self.frequency._hz, repr(self.curve))

    def __str__(self):
        return "<%s: %s, curve = %s>" % (self.__class__.__name__, self.frequency, str(self.curve))


