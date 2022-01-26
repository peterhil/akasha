#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Akasha test assertion utilities.
"""

import numpy as np

from builtins import zip
from numpy.testing import assert_array_equal


def assert_equal_image(expected, actual):
    for row, (pixels_actual, pixels_expected) in enumerate(zip(actual, expected)):
        assert_array_equal(
            pixels_actual.T,
            pixels_expected.T,
            "=== Row %s is not equal ===\n\n"
            "Expected:\n%s\n\n"
            "Actual:\n%s"
            % (row, expected.T, actual.T),
            verbose=True
        )
