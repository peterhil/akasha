#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for curves
"""

import pytest
import numpy as np

from numpy.testing.utils import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.audio.curves import Curve, Circle, Square, Super
from akasha.audio.generators import PeriodicGenerator
from akasha.utils.math import pi2, normalize
from akasha.utils.patterns import Singleton


class TestCurve(object):

    def test_super(self):
        assert issubclass(Curve, PeriodicGenerator)
        assert not issubclass(Curve, Singleton)

    def test_at(self):
        with pytest.raises(NotImplementedError):
            Curve.at(4)

    def test_call(self):
        c = Curve()
        with pytest.raises(NotImplementedError):
            c(4)

    def test_repr(self):
        c = Curve()
        assert 'Curve()' == repr(c)

    def test_str(self):
        c = Curve()
        assert c.__class__.__name__ in str(c)


class TestCircle(object):

    def test_super(self):
        assert issubclass(Circle, Curve)
        assert issubclass(Circle, Singleton)

    def test_at(self):
        c = Circle()
        pts = np.linspace(0, 1.0, 7, endpoint=False)
        assert_nulp_diff(c.at(pts), np.exp(pi2 * 1j * pts), 1)

    def test_at_complex(self):
        c = Circle()
        pts = np.linspace(0, 1 + 1j, 7, endpoint=False)
        assert_nulp_diff(c.at(pts), np.exp(pi2 * 1j * pts), 1)

    def test_at_isperiodic(self):
        assert_nulp_diff(Circle.at(np.arange(-1, 3, 1)), 1 + 0j, 3)

    def test_repr(self):
        o = Circle()
        assert o == eval(repr(o))


class TestSquare(object):

    pts = np.arange(0, 1, 1.0 / 8, dtype=np.float64)

    def test_super(self):
        assert issubclass(Square, Curve)
        assert issubclass(Square, Singleton)

    def test_at(self):
        s = Square()

        octants = np.array([
            +1, +1 + 1j,
            1j, -1 + 1j,
            -1, -1 - 1j,
            -1j, 1 - 1j,
        ], dtype=np.complex128)
        assert_nulp_diff(octants, s.at(self.pts), 2)

    def test_repr(self):
        o = Square()
        assert o == eval(repr(o))


class TestSuper(object):

    pts = np.arange(0, 1, 1.0 / 8, dtype=np.float64)

    def test_super(self):
        assert issubclass(Super, Curve)
        assert not issubclass(Super, Singleton)

    def test_init(self):
        s = Super()
        assert_array_equal(s.superness, np.array([4, 2, 2, 2, 1.0, 1.0], dtype=np.float64))
        assert isinstance(s, Super)

    superness_params = [
        [(),                    (4, 2, 2, 2, 1.0, 1.0)],
        [None,                  (4, 2, 2, 2, 1.0, 1.0)],
        [7,                     (7, 2, 2, 2, 1.0, 1.0)],
        [(4),                   (4, 2, 2, 2, 1.0, 1.0)],
        [(3, 1),                (3, 1, 1, 1, 1.0, 1.0)],
        [(3, 4, 2),             (3, 4, 2, 2, 1.0, 1.0)],
        [(4, 3.1, 2.2, 1.3),    (4, 3.1, 2.2, 1.3, 1.0, 1.0)],
        [(5, 1, 2, 3, 0.5),     (5, 1, 2, 3, 0.5, 0.5)],
        [(1, 2, 3, 4, 5, 6),    (1, 2, 3, 4, 5, 6)],
    ]

    @pytest.mark.parametrize(('args', 'exp'), superness_params)
    def test_get_superness(self, args, exp):
        assert_array_equal(
            np.array(exp, dtype=np.float64),
            Super.get_superness(args),
            verbose=False)

    @pytest.mark.parametrize(('args', 'exp'), superness_params)
    def test_normalise_superness_from_init(self, args, exp):
        assert_array_equal(
            np.array(exp, dtype=np.float64),
            Super(args).superness,
            verbose=False)

    def test_normalise_superness_with_invalid_values(self):
        with pytest.raises(ValueError):
            Super('invalid')

    super_amps = [
        [(4, 2),
            list(np.repeat(1, 8))],
        [(5, 3),
            [1.0000, 1.1025, 1.0579, 1.0169, 1.1225, 1.0169, 1.0579, 1.1025]],
        [(12, 4, 12),
            [1.0000, 2.3784, 1.0000, 2.3784, 1.0000, 2.3784, 1.0000, 2.3784]],
        [(7, 1),
            [1.0000, 0.8504, 0.7654, 0.7210, 0.7071, 0.7210, 0.7654, 0.8504]],
    ]

    @pytest.mark.parametrize(('superness', 'exp'), super_amps)
    def test_at(self, superness, exp):
        s = Super(superness)
        assert_nulp_diff(
            np.round(normalize(exp) * Circle.at(self.pts), 3),
            np.round(s.at(self.pts), 3),
            1
        )

    @pytest.mark.parametrize(('superness', 'exp'), super_amps)
    def test_formula(self, superness, exp):
        assert_nulp_diff(
            exp,
            np.round(Super.formula(self.pts, Super(superness).superness), 4),
            1
        )

    def test_repr(self):
        o = Super()
        assert o == eval(repr(o))
