#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import numbers
import numpy as np
import operator

from cdecimal import Decimal
from copy import copy
from fractions import Fraction


def ops(op):
    """
    Return forward and backward operator functions.
    Usage: __add__, __radd__ = ops(operator.add)

    This function is borrowed and modified from fractions.Fraction.operator_fallbacks(),
    which generates forward and backward operator functions automagically.
    """
    def calc(self, other, cls):
        return cls(op(self, other))

    def forward(self, other):
        cls = self.__class__
        if isinstance(other, type(self.value)):
            return calc(self.value, other, cls)
        elif isinstance(other, self.__class__):
            return calc(self.value, other.value, cls)

        elif isinstance(other, Decimal):
            return calc(Decimal(self.value), Decimal(other), cls)

        elif isinstance(other, numbers.Integral):
            return calc(int(self), int(other), cls)
        elif isinstance(other, numbers.Real):
            return calc(float(self), float(other), cls)
        elif isinstance(other, numbers.Rational):
            return calc(Fraction(self.value), Fraction(other), cls)
        elif isinstance(other, numbers.Complex):
            return calc(complex(self), complex(other), cls)
        # elif isinstance(other, numbers.Number):
        #     return calc(self.value, other, cls)
        else:
            return NotImplemented
    forward.__name__ = '__' + op.__name__ + '__'
    forward.__doc__ = op.__doc__

    def reverse(self, other):
        cls = self.__class__
        if isinstance(other, type(self.value)):
            return calc(other, self.value, cls)
        elif isinstance(other, self.__class__):
            return calc(other.value, self.value, cls)

        elif isinstance(other, Decimal):
            return calc(Decimal(other), Decimal(self.value), cls)

        # elif isinstance(other, numbers.Integral):
        #     return calc(int(other), int(self), cls)
        # elif isinstance(other, numbers.Rational):
        #     return calc(Fraction(other), Fraction(self), cls)
        # elif isinstance(other, numbers.Real):
        #     return calc(float(other), float(self), cls)
        # elif isinstance(other, numbers.Complex):
        #     return calc(complex(other), complex(self), cls)

        elif isinstance(other, numbers.Number):
            return calc(other, self.value, cls)
        else:
            return NotImplemented
    reverse.__name__ = '__r' + op.__name__ + '__'
    reverse.__doc__ = op.__doc__

    return forward, reverse


class NumericUnit(numbers.Number):
    """
    Base numeric unit mixin for automatic arithmetic operations.
    """
    # References
    # ----------
    # PEP 3141 -- A Type Hierarchy for Numbers: http://www.python.org/dev/peps/pep-3141/
    # Python Docs: Data Model -- http://docs.python.org/2/reference/datamodel.html

    @property
    def value(obj):
        return getattr(obj, obj._unit)

    def _normalize_value(self, value):
        return value.value if isinstance(value, type(self)) else value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.value)

    def __str__(self):
        return "<%s: %s %s>" % (self.__class__.__name__, self.value, self._unit.strip('_'))


class ComplexUnit(NumericUnit, numbers.Complex):
    """
    Complex valued unit mixin for automatic arithmetic operations.
    """
    def __complex__(self):
        return complex(self.value)

    def __abs__(self):
        return self.__class__(abs(self.value))

    def __neg__(self):
        return self.__class__(-self.value)

    def __pos__(self):
        return self.__class__(+self.value)

    def conjugate(self):
        return self.__class__(self.value.conjugate())

    def real(self):
        return self.__class__(self.value.real)

    def imag(self):
        return self.__class__(self.value.imag)

    __add__, __radd__ = ops(operator.add)
    __sub__, __rsub__ = ops(operator.sub)
    __mul__, __rmul__ = ops(operator.mul)
    __pow__, __rpow__ = ops(operator.pow)
    __div__, __rdiv__ = ops(operator.div)
    __truediv__, __rtruediv__ = ops(operator.truediv)
    __mod__, __rmod__ = ops(operator.mod)


class RealUnit(ComplexUnit, numbers.Real):
    """
    Real valued unit mixin for automatic arithmetic operations.
    """
    def __float__(self):
        return float(self.value)

    def __trunc__(self):
        return int(float(self.value))

    def __le__(self, other):
        return self.value <= other

    def __lt__(self, other):
        return self.value < other

    __floordiv__, __rfloordiv__ = ops(operator.floordiv)


class RationalUnit(RealUnit, numbers.Rational):
    """
    Rational valued unit mixin for automatic arithmetic operations.
    """
    @property
    def numerator(self):
        return self.value.numerator

    @property
    def denominator(self):
        return self.value.denominator


class IntegralUnit(RationalUnit, numbers.Integral):
    """
    Integral valued unit mixin for automatic arithmetic operations.
    """
    def __long__(self):
        return long(self.value)

    def __int__(self):
        return int(self.value)

    def __invert__(self):
        return ~(self.value)

    __and__, __rand__ = ops(operator.and_)
    __or__, __ror__ = ops(operator.or_)
    __xor__, __rxor__ = ops(operator.xor)

    __lshift__, __rlshift__ = ops(operator.lshift)
    __rshift__, __rrshift__ = ops(operator.rshift)


# Numerical hierarchy
NumericUnit.register(numbers.Complex)
ComplexUnit.register(numbers.Real)
RealUnit.register(numbers.Rational)
RationalUnit.register(numbers.Integral)
