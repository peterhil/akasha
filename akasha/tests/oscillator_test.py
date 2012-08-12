#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for oscillator.py
"""

import pytest
import unittest
import numpy as np

from numpy.testing.utils import assert_array_almost_equal, assert_array_max_ulp
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff
from numpy.testing.utils import nulp_diff

from fractions import Fraction

from ..audio.oscillator import *
from ..utils.math import to_phasor
from ..tunings import cents, cents_diff


# Just noticeable difference allowance for Frequency rounding
# See http://en.wikipedia.org/wiki/Cent_(music)#Human_perception
# This should probably be much smaller than the suggested 3-6 cents...
JND_CENTS_EPSILON = 1.0e-2


class TestOscillatorInit(object):
    """Test oscillator initialization"""

    def test_init(self):
        """Test initialization"""
        o = Osc.from_ratio(1, 8)
        assert isinstance(o, Osc)
        assert o.order == 1
        assert o.period == 8
        assert o.ratio == Fraction(1,8)

    def test_size(self):
        """Test size of roots"""
        assert 8, Osc.from_ratio(1, 8).sample.size

    def test_init_with_aliasing(self):
        sampler.prevent_aliasing = False
        sampler.negative_frequencies = True
        assert Osc.from_ratio(1, 8), Osc.from_ratio(9, 8)
        assert Osc.from_ratio(7, 8), Osc.from_ratio(-1, 8)


class TestOscFrequency(object):
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


class TestOscRoots(object):
    """Test root generating functions."""

    def test_root_func_sanity(self):
        """It should give sane values."""
        wi = 2 * np.pi * 1j
        a = Osc.from_ratio(1, 8).sample
        b = np.array([
            +1+0j, np.exp(wi*1/8),
            +0+1j, np.exp(wi*3/8),
            -1+0j, np.exp(wi*5/8),
            -0-1j, np.exp(wi*7/8),
        ], dtype=np.complex128)
        assert np.allclose(a.real, b.real, atol=1e-13), \
            "real \n%s\nis not close to\n%s" % (a, b)
        assert np.allclose(a.imag, b.imag, atol=1e-13), \
            "imag \n%s\nis not close to\n%s" % (a, b)

        # assert_nulp_diff(a.real, b.real, nulp=1) # @FIXME nulp should be smaller!
        # assert_nulp_diff(a.imag, b.imag, nulp=1)

        assert_nulp_diff(a, b, nulp=1) # complex testing works differently?!

    def test_phasors(self):
        """It should be accurate.
        Uses angles to make testing easier.
        """
        for period in (5, 7, 8, 23): #, 2202):   # + tuple( random.choice(range(1, 44100)) )
            o = Osc.from_ratio(1, period)

            fractional_angle = lambda n: float(Fraction(n, period) % 1) * 360
            angles = np.array( map( fractional_angle, range(0,period) ) )
            angles = 180 - ( (180 - angles) % 360) # wrap 'em to -180..180!

            a = to_phasor(o.sample)
            b = np.array( zip( [1] * period,  angles ) )

            assert_nulp_diff(a.real, b.real, nulp=25) # @FIXME nulp should be smaller!
            assert_nulp_diff(a.imag, b.imag, nulp=1)


class TestOscSlicing(object):
    def setup(self):
        self.o = Osc.from_ratio(1, 6)
        self.p = Osc.from_ratio(3, 8)

    def test_list_access(self):
        self.setup()
        assert self.o[-1] == self.o[5] == self.o[11]
        assert_nulp_diff(self.o[0, 6, 12], self.o[0], 1)

    def test_slice_access(self):
        self.setup()
        assert np.equal(self.p[::], self.p.sample).all()
        assert np.allclose(self.o[:3:2], Osc.from_ratio(1, 3).sample)
        assert np.equal(self.p[-1:8:3], self.p[7,2,5,0,3,6,1,4,7]).all()


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


