#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Mix sound objects together by multiplying the components
"""

import numpy as np
import scipy as sp

from akasha.audio.generators import Generator


class Mix(Generator):
    """
    Mix sound objects together by multiplying the components.
    """

    def __init__(self, *components):
        for component in components:
            if not hasattr(component, 'at'):
                raise RuntimeError("All components need to have at() method!")
        self.components = np.array(components, dtype=object)

    def at(self, t):
        """
        Sample mix at sample times (t).
        """
        return reduce(np.multiply, [component.at(t) for component in self.components])

    def sample(self, frames):
        """
        Sample mix at frames.
        """
        return reduce(np.multiply, [component[frames] for component in self.components])

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
