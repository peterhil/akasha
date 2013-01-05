#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Numeric types
"""

from __future__ import division

import numbers
import operator

from abc import ABCMeta, abstractproperty
from cdecimal import Decimal
from fractions import Fraction


def ops(op):
    """
    Return forward and backward operator functions.
    Usage: __add__, __radd__ = ops(operator.add)

    This function is borrowed and modified from fractions.Fraction.operator_fallbacks(),
    which generates forward and backward operator functions automagically.
    """
    # pylint: disable=C0111,R0911,R0912

    def calc(self, other, cls):
        """
        Closure to calculate the operation with other and return results as instances of cls.
        """
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

        elif isinstance(other, numbers.Number):
            return calc(other, self.value, cls)
        else:
            return NotImplemented
    reverse.__name__ = '__r' + op.__name__ + '__'
    reverse.__doc__ = op.__doc__

    return forward, reverse


class NumericUnit(object):
    """
    Base numeric unit mixin for automatic arithmetic operations.
    """
    # References
    # ----------
    # PEP 3141 -- A Type Hierarchy for Numbers: http://www.python.org/dev/peps/pep-3141/
    # Python Docs: Data Model -- http://docs.python.org/2/reference/datamodel.html

    __metaclass__ = ABCMeta

    @abstractproperty
    def _unit(self):
        """The name of the property as string to use as unit."""
        return NotImplemented

    @property
    def value(self):
        """The value of this numeric unit."""
        return getattr(self, self._unit)

    def _normalize_value(self, value):
        """Prevents type errors by normalising the value to the non-unit value."""
        return value.value if isinstance(value, type(self)) else value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.value)

    def __str__(self):
        return "<%s: %s %s>" % (self.__class__.__name__, self.value, self._unit.strip('_'))


class ComplexUnit(NumericUnit):
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
        """Value of conjugate to self."""
        return self.__class__(self.value.conjugate())

    def real(self):
        """Real value."""
        return self.__class__(self.value.real)

    def imag(self):
        """Imaginary value."""
        return self.__class__(self.value.imag)

    __add__, __radd__ = ops(operator.add)
    __sub__, __rsub__ = ops(operator.sub)
    __mul__, __rmul__ = ops(operator.mul)
    __pow__, __rpow__ = ops(operator.pow)
    __div__, __rdiv__ = ops(operator.div)
    __truediv__, __rtruediv__ = ops(operator.truediv)
    __mod__, __rmod__ = ops(operator.mod)


class RealUnit(ComplexUnit):
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


class RationalUnit(RealUnit):
    """
    Rational valued unit mixin for automatic arithmetic operations.
    """
    @property
    def numerator(self):
        """Numerator property."""
        return self.value.numerator

    @property
    def denominator(self):
        """Deniminator property."""
        return self.value.denominator


class IntegralUnit(RationalUnit):
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


# Numerical hierarchy  pylint: disable=E1101
numbers.Number.register(NumericUnit)
numbers.Complex.register(ComplexUnit)
numbers.Real.register(RealUnit)
numbers.Rational.register(RationalUnit)
numbers.Integral.register(IntegralUnit)
