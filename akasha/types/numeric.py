#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import operator

from numbers import Number


def ops(op):
    """
    Return forward and backward operator functions.
    Usage: __add__, __radd__ = ops(operator.add)

    This function is borrowed and modified from fractions.Fraction.opserator_fallbacks(),
    which generates forward and backward operator functions automagically.
    """
    def calc(self, other):
        return self.__class__(op(self, other))

    def forward(self, other):
        if isinstance(other, self.__class__):
            return calc(self.unit, other.unit)
        elif isinstance(other, (Number, np.number)):
            return calc(self.unit, other)
        else:
            return NotImplemented
    forward.__name__ = '__' + op.__name__ + '__'
    forward.__doc__ = op.__doc__

    def reverse(self, other):
        if isinstance(self, other.__class__):
            return calc(other.unit, self.unit)
        elif isinstance(other, (Number, np.number)):
            return calc(other, self.unit)
        else:
            return NotImplemented
    reverse.__name__ = '__r' + op.__name__ + '__'
    reverse.__doc__ = op.__doc__

    return forward, reverse


class AlgebraicField(object):
    """
    A mixin to enable arithmetic operations with the inherited class.
    Refer to: http://docs.python.org/2/reference/datamodel.html
    """

    @property
    def unit(obj):
        return getattr(obj, obj._unit)

    def __eq__(self, other):
        """Equality"""
        return self.unit == self.__class__(other).unit

    def __hash__(self):
        """Identity hash"""
        return hash(self.unit)

    __add__, __radd__ = ops(operator.add)
    __sub__, __rsub__ = ops(operator.sub)
    __mul__, __rmul__ = ops(operator.mul)
    __div__, __rdiv__ = ops(operator.div)
    __truediv__, __rtruediv__ = ops(operator.truediv)
    __floordiv__, __rfloordiv__ = ops(operator.floordiv)

    __mod__, __rmod__ = ops(operator.mod)
    __pow__, __rpow__ = ops(operator.pow)

    def __pos__(self):
        return self.__class__(self.unit)

    def __neg__(self):
        return self.__class__(-self.unit)

    def __abs__(self):
        return self.__class__(abs(self.unit))

    def __float__(self):
        """Returns a float valued unit."""
        return float(self.unit)

    def __int__(self):
        return int(self.unit)

    def __repr__(self):
        return "%s(%s)" % (self.__class__, self.unit)

    def __str__(self):
        return "<%s: %s %s>" % (self.__class__, self.unit, self.unit.strip('_'))

