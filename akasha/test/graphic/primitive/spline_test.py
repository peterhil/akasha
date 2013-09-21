#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for spline drawing functions
"""

import numpy as np
import pytest

from akasha.graphic.primitive.spline import angle_between, angle_between_dotp, circumcircle_radius
from akasha.audio.curves import Circle
from cmath import rect
from numpy.testing.utils import assert_array_equal


class TestSplines(object):
    """
    Unit tests for spline module.
    """

    octant = np.sqrt(2)/2  # ≈ 0.70710678

    @pytest.mark.parametrize(('a', 'b', 'expected'), [
        [0, 0, 0],
        [1, 1, 0],
        [1j, 2j, 0],
        [-1, 1, np.pi],
        [1j, 1, np.pi / 2],
        [1j, 1 + 1j, np.pi / 4],
        [1 + 2j, 1, 1.1071487177940904],

        # Keep sign?
        [-1j, 1j, -np.pi],
        [octant - octant * 1j, octant + octant * 1j, -np.pi / 2],

        # np.abs(result) > np.pi
        [-1j, -1, -4.7123889803846897],
    ])
    def test_angle_between(self, a, b, expected):
        assert angle_between(a, b) == expected

    def test_circumcircle_radius(self):
        radius = 0.5
        pts = Circle.at(np.linspace(0, 1, 3, endpoint=False)) * radius
        assert apply(circumcircle_radius, pts) == np.abs(radius)
