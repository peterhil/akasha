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

from akasha.types.numeric import AlgebraicField
from cdecimal import Decimal
from fractions import Fraction


class Numeric(AlgebraicField):

    def __init__(self, value):
        self._unit = '_value'
        self._value = self._normalize_value(value)


class TestAlgebraicField(object):
    """Test algebraic field mixin."""

    operations = (
        # Complex
        'add',
        'sub',
        'mul',
        'pow',
        'div',
        'truediv',
        'floordiv',
        'mod',
        # Real
        # 'ge',
        # 'gt',
        # Integral
        # 'and_',
        # 'or_',
        # 'xor',
        # 'lshift',
        # 'rshift',
        )

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
        assert op(cls(a), cls(b)) == op(Numeric(cls(a)), Numeric(cls(b)))  # Self

    @pytest.mark.parametrize(['operation', 'field'], list(itertools.product(operations, types.keys())))
    def test_ops_forward(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == op(Numeric(cls(a)), cls(b))  # Forward

    @pytest.mark.parametrize(['operation', 'field'], list(itertools.product(operations, types.keys())))
    def test_ops_backward(self, operation, field):
        a, b, op, cls = self.op_params(operation, field)
        assert op(cls(a), cls(b)) == op(cls(a), Numeric(cls(b)))  # Backward