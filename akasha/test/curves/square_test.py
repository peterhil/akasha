# C0111: Missing docstring
# R0201: Method could be a function
#
# pylint: disable=C0111,R0201

"""
Unit tests for Square
"""

import numpy as np

from numpy.testing import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.curves import Curve, Square
from akasha.utils.patterns import Singleton


class TestSquare():

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
