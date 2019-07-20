#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for curvature module
"""

import numpy as np
import pytest

from akasha.curves import Circle
from akasha.math.geometry.curvature import estimate_curvature, estimate_curvature_with_ellipses
from numpy.testing.utils import assert_array_almost_equal


class TestEstimateCurvatureWithCircles(object):
    """
    Unit tests for curvature estimation using circumcircle radius.
    """
    curvature_dataset = [
        [Circle.roots_of_unity(4) * 4,        1.0 /  4],
        [Circle.roots_of_unity(5) * 5 + 10,   1.0 /  5],
        [Circle.roots_of_unity(6) * 6 + 30,   1.0 /  6],
        [Circle.roots_of_unity(17) * 17 + 17, 1.0 / 17],
        # Test edge cases on guards
        [[3, 3, 3], np.inf],
        [[1, 2, 3], 0],
    ]

    @pytest.mark.parametrize(('points', 'expected'), curvature_dataset)
    def test_estimate_curvature(self, points, expected):
        assert_array_almost_equal(
            estimate_curvature(points),
            expected
        )

    @pytest.mark.xfail(reason='Five point estimation could work')
    @pytest.mark.parametrize(('points', 'expected'), curvature_dataset)
    def test_estimate_curvature_with_ellipses(self, points, expected):
        assert_array_almost_equal(
            estimate_curvature_with_ellipses(points),
            expected
        )

    @pytest.mark.parametrize(('points', 'expected'), [
        [Circle.roots_of_unity(5) * -2, 0.5],
        [Circle.roots_of_unity(4) * -2, 0.5],
        [Circle.roots_of_unity(6) * -2, 0.5],
    ])
    def test_estimate_curvature_with_negative_radius(self, points, expected):
        assert_array_almost_equal(
            estimate_curvature(points),
            expected
        )

    @pytest.mark.parametrize(('points', 'expected'), [
        [np.flipud(Circle.roots_of_unity(5) * 2), -0.5],
        [np.flipud(Circle.roots_of_unity(4) * 2), -0.5],
        [np.flipud(Circle.roots_of_unity(6) * 2), -0.5],
    ])
    def test_estimate_curvature_with_reversed_points(self, points, expected):
        assert_array_almost_equal(
            estimate_curvature(points),
            expected
        )
