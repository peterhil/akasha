#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Harmonics
"""

import numpy as np

from numpy.testing.utils import assert_array_almost_equal

from akasha.audio.harmonics import Harmonics
from akasha.audio.oscillator import Osc
from akasha.timing import sampler


class TestHarmonics(object):
    """
    Tests for Harmonics
    """

    frames = np.arange(0, sampler.rate, 4410, dtype=np.float64)
    times = frames / float(sampler.rate)

    def test_at(self):
        o = Osc(122.0)
        h = Harmonics(o)
        assert_array_almost_equal(
            h.at(self.times),
            h[self.frames]
        )

    def test_sample_with_iterable(self):
        o = Osc(122.0)
        h = Harmonics(o)
        assert_array_almost_equal(
            h[iter(self.frames)],
            h.at(self.times)
        )
