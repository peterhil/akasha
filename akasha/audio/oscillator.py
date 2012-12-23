#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from numbers import Real

from akasha.audio.curves import Circle
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator


class Osc(FrequencyRatioMixin, PeriodicGenerator):
    """Generic oscillator class with a frequency and a parametric curve."""

    def __init__(self, freq, curve=Circle()):
        super(self.__class__, self).__init__()
        if not isinstance(freq, Real):
            raise TypeError("Argument 'freq' must be a real number.")
        self._hz = Frequency(freq)
        self.curve = curve

    @property
    def sample(self):
        return self.curve.at(Frequency.angles(self.ratio))

    def __repr__(self):
        tpl = "{0}({1}, curve={2})"
        return tpl.format(self.__class__.__name__, self.frequency._hz, repr(self.curve))

    def __str__(self):
        tpl = "<{0}: {1}, curve={2}>"
        return tpl.format(self.__class__.__name__, self.frequency, str(self.curve))
