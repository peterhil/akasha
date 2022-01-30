#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
# R0901: Too many ancestors (n/k)
# C0321: More than one statement on a single line
#
# pylint: disable=C0111,R0201,E1101,R0901,C0321

"""
Unit tests for numeric types
"""

from __future__ import division

import itertools
import operator
import pytest

# from abc import ABCMeta, abstractproperty
from fractions import Fraction

from akasha.types.numeric import \
     NumericUnit, \
     ComplexUnit, \
     RationalUnit, \
     RealUnit, \
     IntegralUnit


class Numeric(NumericUnit):
    # pylint: disable=W0231,R0903

    def __init__(self, value):
        self._value = self._normalize_value(value)

    @property
    def _unit(self):
        return '_value'


class Complex(ComplexUnit, Numeric):
    pass


class Real(RealUnit, Complex):
    pass


class Rational(RationalUnit, Real):
    pass


class Integral(IntegralUnit, Rational):
    pass


complex_operations = [
    'add',
    'sub',
    'mul',
    'truediv',
    'pow',
]

real_operations = complex_operations + [
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


class TestComplexUnit():
    """Test algebraic field mixin."""

    unit = Complex
    operations = complex_operations
    types = {
        'Integral': int,
        'Rational': Fraction,
        'Real': float,
        'Complex': complex,
    }

    def op_params(self, operation, field):
        op = getattr(operator, operation)
        cls = self.types[field]
        return 7, 5, op, cls

    @pytest.mark.parametrize(
        ['operation', 'field'],
        list(itertools.product(operations, types.keys())))
    def test_ops_self(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == \
            op(self.unit(cls(a)), self.unit(cls(b)))  # Self

    @pytest.mark.parametrize(
        ['operation', 'field'],
        list(itertools.product(operations, types.keys())))
    def test_ops_forward(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == \
            op(self.unit(cls(a)), cls(b))  # Forward

    @pytest.mark.parametrize(
        ['operation', 'field'],
        list(itertools.product(operations, types.keys())))
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
