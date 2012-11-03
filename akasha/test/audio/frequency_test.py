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

        # Just noticeable difference allowance for Frequency rounding
        # See http://en.wikipedia.org/wiki/Cent_(music)#Human_perception
        # This should probably be much smaller than the suggested 3-6 cents...
        JND_CENTS_EPSILON = 1.0e-2

        for f in [20.899, 30.0001, 220.0001, 440.899, 2201.34, 8001.378, 12003.989, 20000.1]:
            o = Osc(f)
            assert cents_diff(o.frequency, f) <= JND_CENTS_EPSILON


class TestFrequencyAliasing(object):
    """Test (preventing the) aliasing of frequencies out of range 0 to sample rate."""
    
    silence = Frequency(0)
    negative = Fraction(-1, 7)
    over_nyquist = Fraction(9, 14)
    over_one = Fraction(9, 7)

    @staticmethod
    def freq(ratio):
        return Frequency.from_ratio(ratio)

    def testPreventAliasing(self):
        """It should prevent aliasing when given a ratio out of range 0 to 1/2."""
        sampler.prevent_aliasing = True
        sampler.negative_frequencies = False

        assert self.freq(self.negative) == self.silence
        assert self.freq(self.over_nyquist) == self.silence
        assert self.freq(self.over_one) == self.silence

    def testFrequenciesOverNyquistRate(self):
        """It should not produce silence if ratio > 1/2"""
        sampler.prevent_aliasing = False
        assert self.freq(self.over_nyquist).ratio == self.over_nyquist

    def testFrequenciesOverSamplingRate(self):
        """It should wrap when ratio is over one."""
        sampler.prevent_aliasing = False
        assert self.freq(self.over_one).ratio == self.over_one % 1

    def testNegativeFrequencies(self):
        """It should wrap negative frequencies."""
        sampler.negative_frequencies = True

        sampler.prevent_aliasing = False
        assert self.freq(self.negative).ratio == self.negative % 1

        sampler.prevent_aliasing = True
        assert self.freq(self.negative).ratio == self.negative % 1


