#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for oscillator.py
"""

import unittest
from fractions import Fraction

from ..audio.oscillator import *
from ..utils.math import to_phasor


class OscInitTest(unittest.TestCase):
    """Test oscillator initialization"""

    def testInit(self):
        """Test initialization"""
        o = Osc.from_ratio(1, 8)
        assert(isinstance(o, Osc))
        self.assertEqual(o.order, 1)
        self.assertEqual(o.period, 8)
        self.assertEqual(o.ratio, Fraction(1,8))

    def testSize(self):
        """Test size of roots"""
        self.assertEqual(8, Osc.from_ratio(1, 8).sample.size)

    def testInitWithAliasing(self):
        sampler.prevent_aliasing = False
        sampler.negative_frequencies = True
        self.assertEqual(Osc.from_ratio(1, 8), Osc.from_ratio(9, 8))
        self.assertEqual(Osc.from_ratio(7, 8), Osc.from_ratio(-1, 8))


class OscFrequencyTest(unittest.TestCase):
    """Test frequencies"""

    def testFreqInit(self):
        """It should intialize using a frequency"""
        o = Osc(440)
        assert(isinstance(o, Osc))
        assert(isinstance(o.frequency, Frequency))

    def testFrequency440(self):
        """It should return correct frequency for 440 Hz."""
        o = Osc(440)
        self.assertEqual(o.frequency, 440)

    def testFloatFrequency(self):
        """It should return close frequency from float."""
        o = Osc(440.899)
        self.assertAlmostEqual(o.frequency, 440.899, places=4)


class OscRootsTest(unittest.TestCase):
    """Test root generating functions."""

    def testRootFuncSanity(self):
        """It should give sane values."""
        wi = 2 * np.pi * 1j
        a = Osc.from_ratio(1, 8).sample
        b = np.array([
            +1+0j, np.exp(wi*1/8),
            +0+1j, np.exp(wi*3/8),
            -1+0j, np.exp(wi*5/8),
            -0-1j, np.exp(wi*7/8),
        ])
        assert np.allclose(a.real, b.real, atol=1e-13), \
            "real %s is not close to\n\t\t%s" % (a, b)
        assert np.allclose(a.imag, b.imag, atol=1e-13), \
            "imag %s is not close to\n\t\t%s" % (a, b)

    def testPhasors(self):
        """It should be accurate.
        Uses angles to make testing easier.
        """
        for period in (5,7,8,23):   # + tuple( random.choice(range(1, 44100)) )
            o = Osc.from_ratio(1, period)
            fractional_angle = lambda n: float(Fraction(n, period) % 1) * 360
            angles = np.array( map( fractional_angle, range(0,period) ) )
            angles = 180 - ( (180 - angles) % 360) # wrap 'em to -180..180!
            a = to_phasor(o.sample)
            b = np.array( zip( [1] * period,  angles ) )
            assert np.allclose(a, b), \
                "%s is not close to\n\t\t%s" % (a, b)


class OscSlicingTest(unittest.TestCase):
    def setUp(self):
        self.o = Osc.from_ratio(1, 6)
        self.p = Osc.from_ratio(3, 8)

    def testListAccess(self):
        self.assertEqual(*self.o[-1,5,11])
        self.assertEqual(*self.o[0,6,12])

    def testSliceAccess(self):
        assert np.equal(self.p[::], self.p.sample).all()
        assert np.allclose(self.o[:3:2], Osc.from_ratio(1, 3).sample)
        assert np.equal(self.p[-1:8:3], self.p[7,2,5,0,3,6,1,4,7]).all()


class OscAliasingTest(unittest.TestCase):
    """Test (preventing the) aliasing of frequencies: f < 0, f > sample rate"""
    silence = Osc(0)

    def testFrequenciesOverNyquistRate(self):
        """It should (not) produce silence if ratio > 1/2"""
        sampler.prevent_aliasing = True
        self.assertEqual(Osc.from_ratio(9, 14), self.silence)

        sampler.prevent_aliasing = False
        self.assertEqual(Osc.from_ratio(9, 14).ratio, Fraction(9, 14))

    def testFrequenciesOverSamplingRate(self):
        sampler.prevent_aliasing = True
        self.assertEqual(Osc.from_ratio(9, 7), self.silence)

        sampler.prevent_aliasing = False    # but still wraps when ratio > 1
        self.assertEqual(Osc.from_ratio(9, 7).ratio, Fraction(2, 7))

    def testNegativeFrequencies(self):
        """It should handle negative frequencies according to preferences."""
        sampler.prevent_aliasing = True
        sampler.negative_frequencies = False
        self.assertEqual(Osc.from_ratio(-1, 7), self.silence)

        sampler.prevent_aliasing = False
        sampler.negative_frequencies = True
        self.assertEqual(Osc.from_ratio(-1, 7), Osc.from_ratio(6, 7))

        sampler.prevent_aliasing = True
        sampler.negative_frequencies = True
        ratio = Fraction(-1, 7)
        o = Osc.from_ratio(ratio)
        self.assertEqual(o.frequency, float(ratio * sampler.rate))
        # o.ratio is still modulo sampler.rate - is this right?


if __name__ == "__main__":
    unittest.main()

