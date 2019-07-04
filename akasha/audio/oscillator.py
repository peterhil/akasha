#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The oscillating module.
"""

from __future__ import division

from numbers import Real

from akasha.curves import Circle
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator


class Osc(FrequencyRatioMixin, PeriodicGenerator):
    """
    Generic oscillator which has a closed curve and a frequency.
    """
    def __init__(self, freq, curve=Circle()):
        super(self.__class__, self).__init__()
        if not isinstance(freq, Real):
            raise TypeError("Argument 'freq' must be a real number.")
        self._hz = Frequency(freq)
        self.curve = curve

    def at(self, t):
        """
        Sample oscillator at sample times (t).
        """
        return self.curve.at(self.frequency.at(t))

    @property
    def sample(self):
        """
        Sample one period of the oscillator curve with the current frequency.
        """
        return self.curve.at(Frequency.angles(self.ratio))

    def __repr__(self):
        tpl = "{0}({1}, curve={2})"
        return tpl.format(self.__class__.__name__, self.frequency._hz, repr(self.curve))

    def __str__(self):
        tpl = "<{0}: {1}, curve={2}>"
        return tpl.format(self.__class__.__name__, self.frequency, str(self.curve))
