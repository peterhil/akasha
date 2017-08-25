#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Mixin to create playable composite sound objects
"""

import numpy as np


class BaseFrequencyMixin(object):

    components = []

    def _frequency_components(self):
        """
        Return components which have a frequency.
        """
        return np.array([component for component in self.components if hasattr(component, 'frequency')])

    @property
    def frequency(self):
        """
        Frequency getter.
        """
        return np.min([c.frequency for c in self._frequency_components()])

    @frequency.setter
    def frequency(self, hz):
        """
        Frequency setter.
        """
        old_frequency = np.min(self._frequency_components()).frequency
        change = float(hz) / old_frequency
        for component in self._frequency_components():
            component.frequency *= change
