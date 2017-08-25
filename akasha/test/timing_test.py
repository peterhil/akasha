#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for timing
"""

import numpy as np
import pytest

from numpy.testing.utils import assert_array_almost_equal

from akasha.timing import sampler


class TestSampler(object):
    """Test sampler"""

    @pytest.mark.parametrize(('dtype', 'times'), [
        [np.float64, sampler.slice(8)],
        [np.float64, sampler.slice(7)],
        [np.float64, sampler.slice(1000, step=7)],
        [np.int64, sampler.slice(16)],
        [np.int64, sampler.slice(17)],
    ])
    def test_at(self, dtype, times):
        expected = (times * float(sampler.rate)).astype(dtype)
        assert_array_almost_equal(sampler.at(times, dtype=dtype), expected)

    @pytest.mark.parametrize(('start', 'end'), [
        [0, 8],
        [12, 24],
        [9, None],
    ])
    def test_slice(self, start, end):
        assert_array_almost_equal(
            sampler.slice(start, end) * sampler.rate,
            np.arange(start, end, dtype=np.float64)
        )
