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
from akasha.utils.python import class_name, _super


class Osc(FrequencyRatioMixin, PeriodicGenerator):
    """
    Generic oscillator which has a closed curve and a frequency.
    """
    def __init__(self, freq, curve=Circle()):
        _super(self).__init__()
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
    def cycle(self):
        """
        One cycle of the oscillator curve with the current frequency.
        """
        return self.curve.at(Frequency.angles(self.ratio))

    def __repr__(self):
        return f'{class_name(self)}{(self.frequency._hz!r}, ' + \
            f'curve={self.curve!r})'

    def __str__(self):
        return f'<{class_name(self)}: {self.frequency!s}, ' + \
           f'curve={self.curve!s}>'
