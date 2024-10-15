"""
Sum sound objects together by adding the components together
"""

import numpy as np

from akasha.audio.generators import Generator


class Scalar(Generator):
    """
    Scalar value that can be used with Mix and Sum objects
    """

    def __init__(self, value, dtype=np.float64):
        self.value = np.array(value, dtype=dtype)

    def at(self, t):
        """
        Return scalar value constantly at any time (t).
        """
        return np.repeat(self.value, len(np.asanyarray(t)))
