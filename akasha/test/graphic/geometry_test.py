#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for geometry module
"""

import numpy as np
import pytest

from akasha.curves import Circle, Square
from akasha.graphic.geometry import \
     AffineTransform, \
     angle_between, \
     circumcircle_radius, \
     circumcircle_radius_alt, \
     estimate_curvature, \
     estimate_curvature_with_ellipses, \
     midpoint
from numpy.testing.utils import assert_array_almost_equal
from akasha.utils.math import all_equal


class TestAffineTransform(object):
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
        assert_array_almost_equal(np.diag([0.5, 0.5, 1]), af._matrix)


class TestGeometryFunctions(object):
    """
    Unit tests for geometry functions.
    """
    octant = np.sqrt(2)/2  # ≈ 0.70710678

    @pytest.mark.parametrize(('a', 'b', 'expected'), [
        [0, 0, 0],
        [1, 1, 0],
        [1j, 2j, 0],
        [1, -1, np.pi],
        [1j, 1, -np.pi / 2],
        [1, 0.5 + 0.5j, np.pi / 4],
        [1 + 2j, 1, -1.1071487177940904],
        [2+4j, 1-1j, -1.8925468811915387],

        # Keep sign?
        [-1j, 1j, np.pi],
        [octant - octant * 1j, octant + octant * 1j, np.pi / 2],

        # np.abs(result) > np.pi
        [-1j, -1, -1.5707963267948968],
    ])
    def test_angle_between(self, a, b, expected):
        assert_array_almost_equal(angle_between(a, b), expected)

    @pytest.mark.parametrize(('a', 'b', 'expected'), [
        [-3, 5, 1],
        [1+1j, -1-1j, 0],
        [1.5+1.5j, -1-1j, 0.25+0.25j],
    ])
    def test_midpoint(self, a, b, expected):
        assert midpoint(a, b) == expected


class TestCircumcircleRadius(object):
    """
    Unit tests for circumcircle radius.
    """
    circumcircle_dataset = [
        [Circle.roots_of_unity(5) * 2, 2],
        [Circle.roots_of_unity(4) * 0.25, 0.25],
        # Test edge cases on guards
        [[3, 3, 3], 0],
        [[1, 2, 3], np.inf],
    ]

    @pytest.mark.parametrize(('points', 'expected'), circumcircle_dataset)
    def test_circumcircle_radius(self, points, expected):
        assert_array_almost_equal(circumcircle_radius(*points[:3]), expected)

    @pytest.mark.parametrize(('points', 'expected'), circumcircle_dataset)
    def test_circumcircle_radius_alt(self, points, expected):
        assert_array_almost_equal(circumcircle_radius_alt(*points[:3]), expected)


class TestEstimateCurvatureWithCircles(object):
    """
    Unit tests for curvature estimation using circumcircle radius.
    """
    curvature_dataset = [
        [Circle.roots_of_unity(5) * 2, 0.5],
        [Circle.roots_of_unity(6) * 0.25, 4],
        # Test edge cases on guards
        [[3, 3, 3], np.inf],
        [[1, 2, 3], 0],
    ]

    @pytest.mark.parametrize(('points', 'expected'), curvature_dataset)
    def test_estimate_curvature(self, points, expected):
        assert_array_almost_equal(estimate_curvature(points), expected)

    @pytest.mark.parametrize(('points', 'expected'), curvature_dataset)
    def test_estimate_curvature_with_ellipses(self, points, expected):
        assert_array_almost_equal(estimate_curvature_with_ellipses(points), expected)

    @pytest.mark.parametrize(('points', 'expected'), [
        [Circle.roots_of_unity(5) * -2, -0.5],
        [Circle.roots_of_unity(4) * -2, -0.5],
        [Circle.roots_of_unity(6) * -2, -0.5],
    ])
    def test_estimate_curvature_with_negative_radius(self, points, expected):
        assert_array_almost_equal(estimate_curvature(points), expected)

    @pytest.mark.parametrize(('points', 'expected'), [
        [Circle.roots_of_unity(5) * 2, 0.5],
        [Circle.roots_of_unity(4) * 2, 0.5],
        [Circle.roots_of_unity(6) * -2, -0.5],
    ])
    def test_estimate_curvature_with_reversed_points(self, points, expected):
        assert_array_almost_equal(estimate_curvature(np.flipud(points)), -expected)
