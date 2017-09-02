#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Exponential
"""

import numpy as np
import pytest

from numpy.testing.utils import assert_array_almost_equal
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.audio.envelope import Exponential
from akasha.timing import sampler
from akasha.math import minfloat


class TestExponential(object):
    """
    Tests for Exponential envelope curves.
    """

    def test_init(self):
        rate = -1
        amp = 0.5
        e = Exponential(rate, amp)
        assert rate == e.rate
        assert amp == e.amp

    @pytest.mark.parametrize('rate', [-1, -2, 0, 1, 2, 5])
    def test_exponential_rates(self, rate):
        t = np.linspace(0, 1, 100)
        assert_array_almost_equal(
            np.clip(np.exp(t * rate), 0, 1),
            Exponential(rate, 1.0).at(t)
        )

    half_lifes = [
        (1, -0.69314718055994529),
        (0, np.inf),
        (-0.5, 1.3862943611198906),
        (-1, 0.69314718055994529),
    ]

    @pytest.mark.parametrize(('rate', 'half_life'), half_lifes)
    def test_half_life(self, rate, half_life):
        e = Exponential(rate)
        assert e.half_life == half_life

    @pytest.mark.parametrize(('rate', 'half_life'), half_lifes)
    def test_from_half_life(self, rate, half_life):
        e = Exponential.from_half_life(half_life)
        assert e.rate == rate

    @pytest.mark.parametrize(('rate', 'amp'), [
        # Amp > 1
        (-1, 200),
        (-1, 2),

        # Amp 1
        (-0.5, 1),
        (-1, 1),
        (-2, 1),
        (-128, 1),
        (-2 ** 18 - 1, 1),

        # Amp 0.5
        (-1, 0.5),
        (-128, 0.5),

        # Amp 0.05
        (-1, 0.05),
        (-0.05, 0.05),
        (-2 ** 18 - 1, 0.05),

        # Amp min_float
        (-1, minfloat(1)[0]),
    ])
    def test_scale(self, rate, amp):
        """
        Test that scale reports correct time to reach zero.
        """
        ex = Exponential(rate, amp)
        index = int(ex.scale * sampler.rate)
        window = 50
        msg = ''

        # There should be at least one non-zero item
        non_zero_before_end = False
        end = ex[index - window : index - 1]
        if np.not_equal(0, end).any():
            non_zero_before_end = True
            msg = "too long: Only zeroes found before end."

        # Every item after end should be zero
        all_zero_after_end = False
        after_end = ex[index : index + window]

        if np.equal(0, after_end).all():
            all_zero_after_end = True
            msg = "too short: Not all zero after end."

        assert (all_zero_after_end or non_zero_before_end), \
            "%s\nLength %d %s\nBefore:\n%s\nAfter:\n%s" % (ex, index, msg, end, after_end)

    def test_from_scale(self):
        expected = Exponential(-0.5)
        e = Exponential.from_scale(1490.2664382038824)
        assert expected.rate == e.rate
        assert e.at(e.scale) == 0

    times = sampler.slice(1000, step=100)
    expected_half_scale = np.array([
        1.0, 0.0340716719169156,
        0.0011608788272139, 0.0000395530825361,
        0.0000013476396515, 0.0000000459163361,
        0.0000000015644463, 0.0000000000533033,
        0.0000000000018161, 0.0000000000000619
    ])

    def test_at(self):
        e = Exponential.from_scale(0.5)
        assert e.at(0.51) == 0
        assert_array_almost_equal(
            e.at(self.times),
            self.expected_half_scale
        )

    def test_sample_magnitude(self):
        e = Exponential.from_scale(0.5)
        assert e[sampler.at(0.51)] == 0
        assert e[10] == 0.71324599963206747

    def test_sample(self):
        e = Exponential.from_scale(0.5)
        assert_array_almost_equal(
            e[sampler.at(self.times)],
            self.expected_half_scale
        )

    def test_sample_with_iterable(self):
        e = Exponential.from_scale(0.5)
        assert_array_almost_equal(
            e[iter(sampler.at(self.times))],
            self.expected_half_scale
        )
