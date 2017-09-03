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

from akasha.audio.generators import Generator
from akasha.audio.mixins import Playable


class Mix(Playable, Generator):
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
