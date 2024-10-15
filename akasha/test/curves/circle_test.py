#!/usr/bin/env python
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Circle
"""

import numpy as np

from numpy.testing import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.curves import Curve, Circle
from akasha.math import pi2
from akasha.utils.patterns import Singleton


class TestCircle():

    def test_super(self):
        assert issubclass(Circle, Curve)
        assert issubclass(Circle, Singleton)

    def test_at(self):
        c = Circle()
        pts = np.linspace(0, 1.0, 7, endpoint=False)
        assert_nulp_diff(c.at(pts), np.exp(pi2 * 1j * pts), 1)

    def test_roots_of_unity(self):
        points = np.linspace(0, 1, 5, endpoint=False)
        expected = Circle.at(points)
        assert np.all(Circle.roots_of_unity(5) == expected)

    def test_at_complex(self):
        c = Circle()
        pts = np.linspace(0, 1 + 1j, 7, endpoint=False)
        assert_nulp_diff(c.at(pts), np.exp(pi2 * 1j * pts), 1)

    def test_at_isperiodic(self):
        assert_nulp_diff(Circle.at(np.arange(-1, 3, 1)), 1 + 0j, 3)

    def test_repr(self):
        o = Circle()
        assert o == eval(repr(o))
