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

from akasha.graphic.geometry import AffineTransform, angle_between, circumcircle_radius, midpoint
from akasha.audio.curves import Circle, Square
from numpy.testing.utils import assert_array_almost_equal


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

    def test_circumcircle_radius(self):
        radius = 0.5
        pts = Circle.at(np.linspace(0, 1, 3, endpoint=False)) * radius
        assert_array_almost_equal(apply(circumcircle_radius, pts), np.abs(radius))

    def test_circumcircle_radius_returns_positive_radius(self):
        expected = np.sqrt(2) / 2
        assert circumcircle_radius(1j, 1, 0) == expected
