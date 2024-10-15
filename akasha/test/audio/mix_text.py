# C0111: Missing docstring
# R0201: Method could be a function
#
# pylint: disable=C0111,R0201

"""
Unit tests for Mix
"""

import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal

from akasha.audio.envelope import Exponential
from akasha.audio.harmonics import Harmonics
from akasha.audio.mix import Mix
from akasha.audio.oscillator import Osc
from akasha.audio.overtones import Overtones
from akasha.curves import Super
from akasha.timing import sampler


class TestMix():
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
            Mix(Osc(436), None, [])

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


# TODO Investigate this more
class TestReplacingHarmonics():
    """Test replacing Harmonics class with Mix objects"""

    def test_mix_overtones(self):
        s = Super(3, 3, 3, 3)
        o = Osc(120, curve=s)
        kwargs = dict(
            n=3,
            func=lambda x: x + 1,
            rand_phase=False,
            damping='sine'
        )
        h = Harmonics(o, **kwargs)
        overtones = Overtones(o, **kwargs)
        times = sampler.times(1)

        assert_array_almost_equal(
            h.at(times),
            overtones.at(times)
        )
