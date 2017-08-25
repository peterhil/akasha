#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Sum sound objects together by adding the components together
"""

import numpy as np

from akasha.audio.generators import Generator


class Scalar(Generator):
    """
    Scalar value that can be used with Mix and Sum objects
    """

    def __init__(self, value):
        self.value = np.array(value, dtype=np.complex128)

    def at(self, t):
        """
        Return scalar value constantly at any time (t).
        """
        return np.repeat(self.value, len(np.asanyarray(t)))

    def sample(self, frames):
        """
        Return scalar value constantly at any frame.
        """
        return np.repeat(self.value, len(np.asanyarray(frames)))
