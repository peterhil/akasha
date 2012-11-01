#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for frequency.py
"""

import pytest
import unittest

import numpy as np

from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator
from akasha.audio.oscillator import Osc
from akasha.timing import sampler
from akasha.tunings import cents, cents_diff
from akasha.utils.math import to_phasor, pi2

from fractions import Fraction

from numpy.testing.utils import assert_array_equal, assert_array_almost_equal, assert_array_max_ulp
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff
from numpy.testing.utils import nulp_diff


# Just noticeable difference allowance for Frequency rounding
# See http://en.wikipedia.org/wiki/Cent_(music)#Human_perception
# This should probably be much smaller than the suggested 3-6 cents...
JND_CENTS_EPSILON = 1.0e-2


class TestFrequency(object):
    """Test frequencies"""

    def test_freq_init(self):
        """It should intialize using a frequency"""
        o = Osc(440)
        assert isinstance(o, Osc)
        assert isinstance(o.frequency, Frequency)

    def test_frequency_440(self):
        """It should return correct frequency for 440 Hz."""
        o = Osc(440)
        assert o.frequency == 440

    def test_float_frequency(self):
        """It should return close frequency from float."""
        for f in [20.899, 30.0001, 220.0001, 440.899, 2201.34, 8001.378, 12003.989, 20000.1]:
            o = Osc(f)
            assert cents_diff(o.frequency, f) <= JND_CENTS_EPSILON


class TestOscAliasing(object):
    """Test (preventing the) aliasing of frequencies: f < 0, f > sample rate"""
    silence = Osc(0)

    def testFrequenciesOverNyquistRate(self):
        """It should (not) produce silence if ratio > 1/2"""
        sampler.prevent_aliasing = True
        assert Osc.from_ratio(9, 14) == self.silence

        sampler.prevent_aliasing = False
        assert Osc.from_ratio(9, 14).ratio == Fraction(9, 14)

    def testFrequenciesOverSamplingRate(self):
        sampler.prevent_aliasing = True
        assert Osc.from_ratio(9, 7) == self.silence

        sampler.prevent_aliasing = False    # but still wraps when ratio > 1
        assert Osc.from_ratio(9, 7).ratio == Fraction(2, 7)

    def testNegativeFrequencies(self):
        """It should handle negative frequencies according to preferences."""
        sampler.prevent_aliasing = True
        sampler.negative_frequencies = False
        assert Osc.from_ratio(-1, 7) == self.silence

        sampler.prevent_aliasing = False
        sampler.negative_frequencies = True
        assert Osc.from_ratio(-1, 7) == Osc.from_ratio(6, 7)

        sampler.prevent_aliasing = True
        sampler.negative_frequencies = True
        ratio = Fraction(-1, 7)
        o = Osc.from_ratio(ratio)
        assert o.frequency == float(ratio * sampler.rate) == -6300.0

        # Note! o.ratio is modulo sampler.rate for negative frequencies
        assert o.ratio == Fraction(6, 7)


