#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Akasha test suite.
"""

import numpy as np

from itertools import izip
from numpy.testing.utils import assert_array_equal


def assert_equal_image(expected, actual):
    for row, (pixels_actual, pixels_expected) in enumerate(izip(actual, expected)):
        assert_array_equal(
            pixels_actual.T,
            pixels_expected.T,
            "=== Row %s is not equal ===\n\n"
            "Expected:\n%s\n\n"
            "Actual:\n%s"
            % (row, expected.T, actual.T),
            verbose=True
        )
