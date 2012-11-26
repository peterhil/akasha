#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for numeric types
"""

from __future__ import division

import operator
import pytest

from akasha.types.numeric import AlgebraicField


class Numba(AlgebraicField):

    def __init__(self, value):
        self._unit = '_value'
        self._value = value


class TestAlgebraicField(object):
    """Test algebraic field mixin."""

    def test_arithmetic(self):
        a4 = 440
        a3 = 220
        a2 = 110

        # Forward
        assert Numba(a4) == Numba(a3) + Numba(a3)
        assert Numba(a3) == Numba(a4) - Numba(a3)
        assert Numba(a4) == Numba(a3) * 2
        assert Numba(a3) == Numba(a4) / 2.0
        assert Numba(350) == Numba(700) // 2
        assert Numba(3.5) == operator.truediv(Numba(7), 2)
        assert Numba(350) == operator.floordiv(Numba(701), 2)

        # Backward
        assert Numba(a4) == a3 + Numba(a3)
        assert Numba(a4) == 2.0 * Numba(a3)
        assert Numba(a4) == Numba(a3).__radd__(Numba(a3))


