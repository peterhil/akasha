#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for envelope.py
"""

import unittest
from envelope import *
import math

class ExponentialTest(unittest.TestCase):
    def testInit(self):
        decay = -1
        amp = 0.5
        e = Exponential(decay, amp)
        self.assertEqual(e.decay, decay)
        self.assertEqual(e.amp, amp)
    
    def testExponentialDecay(self):
        e = Exponential(-1, 1.0)
        self.assertEqual(1.0/math.e, e.exponential(44100))
        f = Exponential(-2, 1.0)
        self.assertEqual(1.0/math.e**2, f.exponential(44100))
    
    def testExponentialGrowth(self):
        e = Exponential(1, 1.0)
        self.assertEqual(math.e, e.exponential(44100))
        f = Exponential(5, 1.0)
        self.assertAlmostEqual(math.e**5, f.exponential(44100))
    
    def testZeroExponential(self):
        amp = 1.0
        z = Exponential(0, amp)
        assert np.equal(amp, z[:44100]).all()


if __name__ == "__main__":
    unittest.main()