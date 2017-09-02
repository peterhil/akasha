#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for ADSR envelopes
"""

import numpy as np
import pytest

from numpy.testing.utils import assert_array_almost_equal

from akasha.audio import Delay, Scalar, Sum
from akasha.audio.envelope import Adsr, Beta, InverseBeta


class TestAdsr(object):
    """
    Tests for ADSR envelopes.
    """

    def test_init(self):
        adsr = Adsr(attack=(0.125), decay=(1.5), sustain=0.6, release=(2), released_at=0.32, decay_overlap=0.25)
        assert isinstance(adsr.attack, Beta)
        assert isinstance(adsr.decay, Delay)
        assert isinstance(adsr.decay.sound, InverseBeta)
        assert isinstance(adsr.release, InverseBeta)
        assert isinstance(adsr.sustain, Scalar)
        assert adsr.attack.time == 0.125
        assert adsr.decay.sound.time == 1.5  # FIXME The API breaks the law of demeter
        assert adsr.release.time == 2.0
        assert adsr.sustain_level == 0.6
        assert adsr.decay_overlap == 0.25
        assert adsr.released_at == 0.32

    def test_decay(self):
        pass

    def test_sustain_level(self):
        adsr = Adsr(sustain=0.9)
        assert adsr.sustain_level == 0.9

    def test_release_at(self):
        adsr = Adsr()
        assert adsr.released_at is None
        adsr.release_at(0.3)
        assert adsr.released_at == 0.3
        with pytest.raises(ValueError):
            adsr.release_at(None)

    def test_at(self):
        adsr = Adsr(attack=(0.5, 1, 1), decay=(0.5, 1, 1), sustain=0.5, release=(0.5, 1, 1), released_at=2.0, decay_overlap=-0.5)
        times = np.linspace(-0.5, 3, 8, endpoint=True)
        assert_array_almost_equal(
            adsr.at(times),
            np.array([ 0. ,  0. ,  1. ,  1. ,  0.5,  0.5,  0. ,  0. ])
        )
