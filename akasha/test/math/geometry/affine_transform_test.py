#!/usr/bin/env python
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for AffineTransform class
"""

import numpy as np

from akasha.curves import Square
from akasha.math.geometry.affine_transform import AffineTransform
from numpy.testing import assert_array_almost_equal


class TestAffineTransform():
    """
    Unit tests for AffineTransform class.
    """
    def test_affine_transform(self):
        sq = Square.at(np.arange(0.125, 1.125, 0.25 / 2))
        scale = (0.5, 0.5)
        af = AffineTransform(scale=scale)
        assert np.all(sq * 0.5 == af(sq))

    def test_affine_estimate(self):
        sq = Square.at(np.arange(0.125, 1.125, 0.25 / 2))
        af = AffineTransform()
        af.estimate(sq, sq * 0.5)
        assert_array_almost_equal(np.diag([0.5, 0.5, 1]), af.params)
