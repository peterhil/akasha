#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from akasha.audio.curves import Circle
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator


class Osc(FrequencyRatioMixin, PeriodicGenerator):
    """Generic oscillator class with a frequency and a parametric curve."""

    def __init__(self, freq, curve=Circle()):
        self._hz = Frequency(freq)
        self.curve = curve

    @property
    def sample(self):
        return self.curve.at(Frequency.angles(self.ratio))

    def __repr__(self):
        return "%s(%s, curve=%s)" % (self.__class__.__name__, self.frequency._hz, repr(self.curve))

    def __str__(self):
        return "<%s: %s, curve=%s>" % (self.__class__.__name__, self.frequency, str(self.curve))
