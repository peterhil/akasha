#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for mathematical functions
"""

import pytest
import numpy as np

from numpy.testing.utils import assert_array_equal

from akasha.funct import pairwise
from akasha.math import *


class TestPrimes(object):
    """Test prime functions."""

    @pytest.mark.parametrize(('interval', 'expected'), [
        [
            [0, 100],
            [
                2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31, 37,  41,
                43, 47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,
            ],
        ],
        [
            [100, 350],
            [
                101, 103, 107,
                109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
                269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
            ],
        ]
    ])
    def test_primes(self, interval, expected):
        assert_array_equal(
            np.array(expected, dtype=np.uint64),
            primes(*interval)
        )

    @pytest.mark.slow
    def test_primes_peak_to_peak(self):
        assert_array_equal(
            np.array([19, 32, 26, 26, 28, 30, 28, 28, 32], dtype=np.uint64),
            np.array([
                np.ptp(distances(primes(i, j)))
                for i, j in pairwise(np.arange(0, 10000, 1000))
            ])
        )

    def test_primes_from_nonzero_lower_limit(self):
        assert_array_equal(
            np.array([7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], dtype=np.uint64),
            primes(7, 47)
        )
