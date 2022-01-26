#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Frequency
"""

import abc
import numbers
import numpy as np
import operator
import pytest

from fractions import Fraction
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import PeriodicGenerator, Generator
from akasha.audio.oscillator import Osc
from akasha.timing import sampler
from akasha.types.numeric import NumericUnit, ComplexUnit, RealUnit
from akasha.math import cents_diff


class TestFrequencyRatioMixin():
    """
    Test Frequency ratios.
    """

    silence = 0
    a4 = 440.0
    a3 = 220.0
    a2 = 110.0
    ratio = Fraction(1, 6)
    hz = float(ratio * sampler.rate)

    hz_types = [
        [Frequency, float],
        [Osc, Frequency]
    ]

    @classmethod
    def setup_class(cls):
        assert issubclass(Frequency, FrequencyRatioMixin)

    def test_from_ratio(self):
        """It should initialize correctly from ratio."""
        f = Frequency.from_ratio(self.ratio.numerator, self.ratio.denominator)
        assert isinstance(f, Frequency)
        assert self.ratio == f.ratio

        f2 = Frequency.from_ratio(self.ratio)
        assert isinstance(f2, Frequency)
        assert self.ratio == f2.ratio

    @pytest.mark.parametrize(('cls', 'hz'), hz_types)
    def test_frequency(self, cls, hz):
        o = cls(self.a4)
        assert self.a4 == o.frequency
        assert isinstance(o.frequency, hz)

    @pytest.mark.parametrize(('cls', 'hz'), hz_types)
    def test_frequency_setter(self, cls, hz):
        o = cls(self.a4)
        o.frequency = self.a2
        assert self.a2 == o.frequency
        assert isinstance(o.frequency, hz)

    @pytest.mark.parametrize(('cls',), [[Frequency], [Osc]])
    def test_ratio(self, cls):
        o = cls(self.a4)
        assert Frequency(self.a4).ratio == o.ratio
        assert isinstance(o.ratio, Fraction)

        o.frequency = self.a2
        assert Frequency(self.a2).ratio == o.ratio
        assert isinstance(o.ratio, Fraction)

    def test_hz(self):
        assert self.hz == Frequency(self.hz).hz

    def test_period_and_order(self):
        f = Frequency(self.ratio)
        assert Frequency(self.ratio) == Frequency(f.order, f.period)

    def test_nonzero(self):
        assert Frequency(self.a4)
        assert Frequency(self.silence) == False

    def test_cmp(self):
        """It should compare correctly."""
        assert Frequency(self.a4) == Frequency(self.a4) > Frequency(self.a2) < Frequency(self.a3)
        assert Frequency(self.a4) == Osc(self.a4) > Osc(self.a2) < Osc(self.a3)
        assert Frequency(self.a4) == self.a4

        assert Frequency(self.a4) > Frequency(self.a2)
        assert Frequency(self.a4) > Osc(self.a2)
        assert Frequency(self.a4) > self.a2

        assert Frequency(self.a2) < Frequency(self.a4)
        assert Frequency(self.a2) < Osc(self.a4)
        assert Frequency(self.a2) < self.a4

    def test_float(self):
        assert self.hz == float(Frequency(self.hz))

    def test_int(self):
        assert int(self.hz) == int(Frequency(self.hz))


class TestFrequencyAliasing():
    """
    Test (preventing the) aliasing of frequencies out of range 0 to sample rate.
    """
    silence = Frequency(0)
    negative = Fraction(-1, 7)
    over_nyquist = Fraction(9, 14)
    over_one = Fraction(9, 7)

    @staticmethod
    def freq(ratio):
        return Frequency.from_ratio(ratio)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in log2")
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


class TestFrequency():
    """
    Test frequencies
    """

    def test_mro(self):
        assert [
            Frequency,
            FrequencyRatioMixin,
            RealUnit,
            ComplexUnit,
            NumericUnit,
            PeriodicGenerator,
            Generator,
            object
        ] == Frequency.mro()

    def test_meta(self):
        assert issubclass(Frequency, numbers.Real)
        assert Frequency.__metaclass__ ==  abc.ABCMeta

    def test_class(self):
        assert issubclass(Frequency, RealUnit)
        assert issubclass(Frequency, FrequencyRatioMixin)
        assert issubclass(Frequency, PeriodicGenerator)
        assert issubclass(Frequency, object)

    def test_init(self):
        """It should intialize using a frequency"""
        # pylint: disable=W0212
        hz = 440.001
        f = Frequency(hz)
        assert isinstance(f.frequency, float)
        assert isinstance(f._hz, float)
        assert hz == f == f.frequency
        assert hz == f._hz

    @pytest.mark.parametrize(('hz',), [
        [20.899],
        [30.0001],
        [220.0001],
        [440.899],
        [2201.34],
        [8001.378],
        [12003.989],
        [20000.1]
    ])
    def test_rounding_frequencies(self, hz):
        """It should not exceed the just noticeable difference for hearing of frequencies."""
        # Just noticeable difference allowance for Frequency rounding
        # See http://en.wikipedia.org/wiki/Cent_(music)#Human_perception
        # This should probably be much smaller than the suggested 3-6 cents...
        JND_CENTS_EPSILON = 1.0e-02
        assert cents_diff(hz, Frequency(hz)) <= JND_CENTS_EPSILON

    @pytest.mark.parametrize(('ratio'), [
        (Fraction(22, 2205)),  # 440 Hz
        (Fraction(-1, 7)),     # Negative
        (Fraction(9, 14)),     # Over Nyquist
        (Fraction(9, 7))       # Over one
    ])
    def test_ratio(self, ratio):
        """It should not wrap and antialias ratio when unwrapped."""
        f = Frequency.from_ratio(ratio, unwrapped=True)
        assert ratio == f.ratio

    def test_angles(self):
        """Is should calculate the frequency angles correctly."""
        assert np.array([0.]) == Frequency.angles(0)
        assert_nulp_diff(
            np.array([0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], dtype=np.float64),
            Frequency.angles(Fraction(1, 8)),
            1
        )

    def test_cycle(self):
        ratio = Fraction(3, 17)
        assert_array_equal(
            Frequency.angles(ratio),
            Frequency.from_ratio(ratio).cycle
        )

    def test_repr(self):
        f = Frequency(100)
        assert f == eval(repr(f))

    def test_str(self):
        s = str(Frequency(100))
        assert 'Frequency' in s
        assert '100' in s

    def test_int(self):
        assert 21 == int(Frequency(21.5))

    @pytest.mark.filterwarnings("ignore:invalid value encountered in log2")
    def test_pos_neg_abs(self):
        sampler.negative_frequencies = True
        hz = 440
        assert hz == +Frequency(hz)
        assert -hz == -Frequency(hz)
        assert abs(hz) == abs(-Frequency(hz))

    def test_arithmetic(self):
        a4 = 440
        a3 = 220

        # Forward
        assert Frequency(a4) == Frequency(a3) + Frequency(a3)
        assert Frequency(a3) == Frequency(a4) - Frequency(a3)
        assert Frequency(a4) == Frequency(a3) * 2
        assert Frequency(a3) == Frequency(a4) / 2.0
        assert Frequency(350) == Frequency(700) // 2
        assert Frequency(3.5) == operator.truediv(Frequency(7), 2)
        assert Frequency(350) == operator.floordiv(Frequency(701), 2)

        # Backward
        assert Frequency(a4) == a3 + Frequency(a3)
        assert Frequency(a4) == 2.0 * Frequency(a3)
        assert Frequency(a4) == Frequency(a3).__radd__(Frequency(a3))
