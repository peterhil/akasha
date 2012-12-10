#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for numeric types
"""

from __future__ import division

import itertools
import numpy as np
import operator
import pytest

from akasha.types.numeric import NumericUnit, ComplexUnit, RationalUnit, RealUnit, IntegralUnit
from cdecimal import Decimal
from fractions import Fraction


class NumericUnit(NumericUnit):
    def __init__(self, value):
        self._unit = '_value'
        self._value = self._normalize_value(value)

class Complex(ComplexUnit, NumericUnit): pass
class Real(RealUnit, Complex): pass
class Rational(RationalUnit, Real): pass
class Integral(IntegralUnit, Rational): pass


complex_operations = [
    'add',
    'sub',
    'mul',
    'div',
    'pow',
    ]

real_operations = complex_operations + [
    'truediv',
    'floordiv',
    'mod',
    'ge',
    'gt',
    ]

integral_operations = real_operations + [
    'and_',
    'or_',
    'xor',
    'lshift',
    'rshift',
    ]


class TestComplexUnit(object):
    """Test algebraic field mixin."""

    unit = Complex
    operations = complex_operations
    types = {
        'Integral': int,
        'Rational': Fraction,
        'Real': float,
        'Complex': complex,
        'Number': Decimal,
        }

    def op_params(self, operation, field):
        op = getattr(operator, operation)
        cls = self.types[field]
        return 7, 5, op, cls

    @pytest.mark.parametrize(['operation', 'field'], list(itertools.product(operations, types.keys())))
    def test_ops_self(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == op(self.unit(cls(a)), self.unit(cls(b)))  # Self

    @pytest.mark.parametrize(['operation', 'field'], list(itertools.product(operations, types.keys())))
    def test_ops_forward(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == op(self.unit(cls(a)), cls(b))  # Forward

    @pytest.mark.parametrize(['operation', 'field'], list(itertools.product(operations, types.keys())))
    def test_ops_backward(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == op(cls(a), self.unit(cls(b)))  # Backward


class TestRealUnit(TestComplexUnit):
    """Test real valued algebraic unit."""

    unit = Real
    operations = real_operations


class TestRationalUnit(TestRealUnit):
    """Test real valued algebraic unit."""

    unit = Rational
    operations = real_operations


class TestIntegralUnit(TestRationalUnit):
    """Test integral valued algebraic unit."""

    unit = Integral
    operations = integral_operations

