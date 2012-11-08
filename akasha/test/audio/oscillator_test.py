#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for oscillator.py
"""

import pytest
import unittest

import numpy as np

from akasha.audio.curves import Circle, Curve
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator
from akasha.audio.oscillator import Osc
from akasha.timing import sampler, time_slice
from akasha.tunings import cents_diff
from akasha.utils.math import to_phasor, pi2

from fractions import Fraction
from numpy.testing.utils import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff


class TestOscillator(object):
    """Test oscillator"""

    def test_class(self):
        assert issubclass(Osc, FrequencyRatioMixin)
        assert issubclass(Osc, PeriodicGenerator)
        assert issubclass(Osc, object)

    def test_init(self):
        a = Osc(440)
        assert isinstance(a, Osc)
        assert a.frequency == Frequency(440.0)
        assert isinstance(a.curve, Circle)
        assert callable(a.curve)

        b = Osc(216, Curve)
        with pytest.raises(NotImplementedError):
            b.curve.at(4)

    def test_from_ratio(self):
        o, p = 3, 802
        a = Osc.from_ratio(o, p)
        b = Osc.from_ratio(Fraction(o, p))
        c = Osc(Fraction(o, p) * sampler.rate)
        assert a == b == c
        assert a.order == o
        assert a.period == p
        assert a.ratio == Fraction(o, p)

    def test_sample(self):
        o, p = 1, sampler.rate

        expected = np.exp(1j * pi2 * o * np.arange(0, 1.0, 1.0/p, dtype=np.float64))

        assert_nulp_diff(
            Osc.from_ratio(o, p).sample,
            expected,
            1
        )

    def test_sample_period_is_accurate(self):
        o = Osc(1)
        s = sampler.rate
        assert_array_equal(o[0*s:1*s], o[1*s:2*s])
        assert_array_equal(o[0*s:1*s], o[2*s:3*s])

    def test_str(self):
        o = Osc(100)
        assert 'Osc' in str(o)

    def test_repr(self):
        o = Osc(100)
        assert o == eval(repr(o))


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
        for period in (5, 7, 8, 23): #, 2202):
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
        assert_array_equal(self.p[7,2,5,0,3,6,1,4,7], self.p[-1:8:3])


