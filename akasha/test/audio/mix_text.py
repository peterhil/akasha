#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Mix
"""

import numpy as np
import pytest

from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from akasha.audio.envelope import Exponential
from akasha.audio.harmonics import Overtones
from akasha.audio.mix import Mix
from akasha.audio.oscillator import Osc
from akasha.audio.scalar import Scalar
from akasha.audio.sum import Sum
from akasha.curves import Super
from akasha.timing import sampler


class TestMix(object):
    """Test mixing sound objects"""

    def get_fixture(self, f1=345.6, f2=436, rate=-1):
        o = Osc(f1)
        o2 = Osc(f2)
        e = Exponential(rate)
        m = Mix(o, o2, e)
        return [o, o2, e, m]

    def test_init(self):
        o = Osc(512)
        e = Exponential(-0.6)
        m = Mix(o, e)
        assert isinstance(m, Mix)

    def test_init_assertion(self):
        with pytest.raises(RuntimeError):
            m = Mix(Osc(436), None, [])

    def test_at(self):
        times = sampler.slice(10000) * sampler.rate
        [o, o2, e, m] = self.get_fixture()
        assert_array_equal(
            m.at(times),
            o.at(times) * o2.at(times) * e.at(times)
        )

    def test_sample(self):
        times = sampler.slice(10000)
        [o, o2, e, m] = self.get_fixture()
        assert_array_equal(
            m[times],
            o[times] * o2[times] * e[times]
        )

    def test_frequency_components(self):
        [o, o2, e, m] = self.get_fixture()
        assert_array_equal(m._frequency_components(), [o, o2])

    def test_frequency(self):
        [o, o2, e, m] = self.get_fixture(120, 330)
        assert m.frequency == 120

    def test_frequency_setter(self):
        [o, o2, e, m] = self.get_fixture(120, 330)
        m.frequency = 420
        assert m.frequency == 420


class TestReplacingOvertones(object):
    """Test replacing Overtones class with Mix objects"""

    def test_mix_overtones(self):
        s = Super(3, 3, 3, 3)
        o = Osc(120, curve=s)
        h = Overtones(o, n=3, func=lambda x: x + 1, rand_phase=False, damping='sine')

        ## Mix generated overtones
        # TODO: Move this part out of test code, and replace Overtones class!
        from itertools import izip
        from akasha.math import map_array, random_phasor
        overtones = map_array(h.func, np.arange(h.n))
        base = s
        # Parts
        frequencies = o.frequency * overtones
        oscs = [Osc(f, base) for f in frequencies]
        envelopes = [Exponential(h.damping(f)) for f in frequencies]
        if h.rand_phase:
            phases = random_phasor(h.n)
        else:
            phases = np.repeat(1.0, h.n).astype(np.complex128)
        phases = [Scalar(phase) for phase in phases]
        parts = [Mix(*part) for part in izip(oscs, envelopes)]
        mix = Sum(*[Mix(*part) for part in izip(parts, phases)])
        ## End Mix generated overtones

        assert_array_equal(
            h[sampler.slice(100) * sampler.rate],
            mix[sampler.slice(100) * sampler.rate]
        )
