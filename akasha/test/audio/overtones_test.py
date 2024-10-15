#!/usr/bin/env python
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Overtones
"""

from numpy.testing import assert_array_almost_equal

from akasha.audio.mix import Mix
from akasha.audio.oscillator import Osc
from akasha.audio.overtones import Overtones
from akasha.audio.sum import Sum
from akasha.audio.scalar import Scalar
from akasha.timing import sampler


class TestOvertones():
    """
    Tests for Overtones
    """
    def test_init(self):
        pass

    #
    # Feature tests
    #
    def test_unit(self):
        times = sampler.times(1)
        osc = Osc(436)
        overtones = Overtones(osc, n=1)
        assert_array_almost_equal(
            overtones.at(times),
            osc.at(times)
        )

    def test_intervals(self):
        # TODO Parametrize or test just the overtone generating function
        freq = 436.0
        ratio = 2.0

        times = sampler.times(1)
        osc = Osc(freq)
        oscs = Mix(Sum(osc, Osc(freq * ratio)), Scalar(0.5))
        overtones = Overtones(osc, n=2, func=lambda f: f + 1)
        assert_array_almost_equal(
            overtones.at(times),
            oscs.at(times)
        )
