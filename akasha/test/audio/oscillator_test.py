#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for oscillator.py
"""

import pytest

import numpy as np

from akasha.audio.curves import Circle, Curve
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator
from akasha.audio.oscillator import Osc
from akasha.timing import sampler
from akasha.utils.math import map_array, to_phasor, pi2

from fractions import Fraction
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

    def test_sample(self):
        o, p = 1, sampler.rate
        expected = np.exp(1j * pi2 * o * np.arange(0, 1.0, 1.0 / p, dtype=np.float64))
        assert_nulp_diff(
            Osc.from_ratio(o, p).sample,
            expected,
            1
        )

    def test_str(self):
        o = Osc(100)
        assert 'Osc' in str(o)

    def test_repr(self):
        o = Osc(100)
        assert o == eval(repr(o))


class TestOscRoots(object):
    """Test root generating functions."""

    def test_root_func_sanity(self):
        """It should give sane values."""
        wi = 2 * np.pi * 1j
        a = Osc.from_ratio(1, 8).sample
        b = np.array([
            +1 + 0j, np.exp(wi * 1 / 8),
            +0 + 1j, np.exp(wi * 3 / 8),
            -1 + 0j, np.exp(wi * 5 / 8),
            -0 - 1j, np.exp(wi * 7 / 8),
        ], dtype=np.complex128)

        assert np.allclose(a.real, b.real, atol=1e-13), \
            "real \n%s\nis not close to\n%s" % (a, b)
        assert np.allclose(a.imag, b.imag, atol=1e-13), \
            "imag \n%s\nis not close to\n%s" % (a, b)

        assert_nulp_diff(a, b, nulp=1)

    def test_phasors(self):
        """It should be accurate.
        Uses angles to make testing easier.
        """
        for period in (5, 7, 8, 23):
            o = Osc.from_ratio(1, period)

            fractional_angle = lambda n: float(Fraction(n, period) % 1) * 360
            angles = map_array(fractional_angle, np.arange(0, period), method='vec')
            angles = 180 - ((180 - angles) % 360)  # wrap 'em to -180..180!

            a = to_phasor(o.sample)
            b = np.array(zip([1] * period, angles))

            assert_nulp_diff(a.real, b.real, nulp=25)  # FIXME: nulp should be smaller!
            assert_nulp_diff(a.imag, b.imag, nulp=1)
