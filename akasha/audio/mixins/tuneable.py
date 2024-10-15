#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""Tuneable mixin module"""

import numpy as np

from akasha.audio.mixins.composite import Composite


class Tuneable(Composite):
    """Mixin to create playable composite sound object that has a
    tuneable frequency.
    """

    def _frequency_components(self):
        """Return components which have a frequency."""
        return self._components_with_attribute('frequency')

    @property
    def frequency(self):
        """Frequency getter."""
        return np.min([c.frequency for c in self._frequency_components()])

    @frequency.setter
    def frequency(self, hz):
        """Frequency setter."""
        old_frequency = np.min(self._frequency_components()).frequency
        change = float(hz) / old_frequency
        for component in self._frequency_components():
            component.frequency *= change
