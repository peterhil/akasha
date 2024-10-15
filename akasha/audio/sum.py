#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Sum sound objects together by adding the components together
"""

import numpy as np

from akasha.audio.generators import Generator
from akasha.audio.mixins import Playable
from akasha.math.functions import summation


class Sum(Playable, Generator):
    """
    Summation of sound objects.
    """

    def __init__(self, *components):
        for component in components:
            if not hasattr(component, 'at'):
                raise RuntimeError("All components need to have at() method!")
        self.components = np.array(components, dtype=object)

    def at(self, t):
        """
        Sample summation at sample times (t).
        """
        return summation([component.at(t) for component in self.components])

    def sample(self, frames):
        """
        Sample summation at frames.
        """
        return summation([component[frames] for component in self.components])
