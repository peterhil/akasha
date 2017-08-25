#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Overtones
"""

import numpy as np

from numpy.testing.utils import assert_array_almost_equal

from akasha.audio.oscillator import Osc
from akasha.audio.harmonics import Overtones
from akasha.timing import sampler


class TestOvertones(object):
    """
    Tests for Overtones
    """

    def test_at(self):
        times = np.linspace(0, 1, 100)
        o = Osc(122.0)
        h = Overtones(o)
        assert_array_almost_equal(h.at(times), h[times])

    def test_sample_with_iterable(self):
        o = Osc(122.0)
        h = Overtones(o)
        times = sampler.slice(sampler.rate, step=1000)
        expected = np.array([1.0,  0.204477,  0.041811,  0.008549,  0.001748])
        assert_array_almost_equal(
            h[iter(times)],
            expected
        )
