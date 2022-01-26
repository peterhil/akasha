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

from akasha.curves import Circle
from akasha.math.geometry import angle_between, circumcircle_radius, circumcircle_radius_alt, midpoint
from numpy.testing import assert_array_almost_equal
from akasha.math import all_equal


class TestGeometryFunctions():
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


class TestCircumcircleRadius():
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
