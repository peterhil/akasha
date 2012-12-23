#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for Exponential
"""

import unittest

import math
import numpy as np

from akasha.audio.envelope import Exponential
from akasha.utils.math import minfloat


class ExponentialTest(unittest.TestCase):
    def testInit(self):
        rate = -1
        amp = 0.5
        e = Exponential(rate, amp)
        self.assertEqual(e.rate, rate)
        self.assertEqual(e.amp, amp)

    def testExponentialDecay(self):
        e = Exponential(-1, 1.0)
        self.assertEqual(1.0 / math.e, e.sample(44100))
        f = Exponential(-2, 1.0)
        self.assertEqual(1.0 / math.e ** 2, f.sample(44100))

    def testExponentialGrowth(self):
        e = Exponential(1, 1.0)
        self.assertEqual(math.e, e.sample(44100))
        f = Exponential(5, 1.0)
        self.assertAlmostEqual(math.e ** 5, f.sample(44100))

    def testZeroExponential(self):
        amp = 1.0
        z = Exponential(0, amp)
        assert np.equal(amp, z[:44100]).all()

    def test_scale(self):
        """Test that scale reports correct time for reach zero."""
        min_float = minfloat(1)[0]
        window = 50
        testdata = [
            # Amp > 1
            (-1, 200), (-1, 2),
            # Amp 1
            (-0.5, 1), (-1, 1), (-2, 1), (-128, 1), (-2 ** 18 - 1, 1),
            # Amp 0.5
            (-1, 0.5), (-128, 0.5),
            # Amp 0.05
            (-1, 0.05), (-0.05, 0.05), (-2 ** 18 - 1, 0.05),
            # Amp min_float
            (-1, min_float),
        ]
        for rate, amp in testdata:
            e = Exponential(rate, amp)
            i = int(e.scale)

            # There should be at least one non-zero item
            non_zero_before_end = False
            end = e[i - window:i - 1]
            if np.not_equal(0, end).any():
                non_zero_before_end = True
                msg = "too long: Only zeroes found before end."

            # Every item after end should be zero
            all_zero_after_end = False
            after_end = e[i:i + window]
            if np.equal(0, after_end).all():
                all_zero_after_end = True
                msg = "too short: Not all zero after end."

            self.assert_(
                (all_zero_after_end or non_zero_before_end),
                "%s\nLength %d %s\nBefore:\n%s\nAfter:\n%s" % (e, i, msg, end, after_end)
            )


if __name__ == '__main__':
    unittest.main()
