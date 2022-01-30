#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Akasha test assertion utilities.
"""

import numpy as np

from builtins import zip
from numpy.testing import assert_array_equal


def assert_equal_image(expected, actual):
    for row, (pixels_actual, pixels_expected) in \
      enumerate(zip(actual, expected)):
        assert_array_equal(
            pixels_actual.T,
            pixels_expected.T,
            f"=== Row {row} is not equal ===\n\n"
            f"Expected:\n{expected.T}\n\n"
            f"Actual:\n{actual.T}",
            verbose=True
        )
