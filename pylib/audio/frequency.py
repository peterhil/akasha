#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import operator

from fractions import Fraction
from numbers import Number

from audio.generators import PeriodicGenerator

from timing import Sampler

from utils.decorators import memoized
from utils.log import logger
from utils.math import *


class FrequencyRatioMixin:
    @property
    def frequency(self):
        return self._hz

    @frequency.setter
    def frequency(self, hz):
        if isinstance(hz, Frequency):
            self._hz = hz
        else:
            self._hz = Frequency(hz)  # Use Trellis, and make a interface for frequencies

    @property
    def ratio(self):
        return self._hz.ratio

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator


class Frequency(object, FrequencyRatioMixin, PeriodicGenerator):
    """Frequency class"""

    def __init__(self, hz):
        # Original frequency, independent of sampling rate or optimizations
        self._hz = float(hz)

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * Sampler.rate)

    @property
    def ratio(self):
        return self.wrap(self.antialias(self.to_ratio(self._hz)))

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * Sampler.rate)

    @staticmethod
    @memoized
    def angles(ratio):
        """Fastest generating method so far. Could be made still faster by using conjugate for half of the samples."""
        if ratio == 0:
            return np.array([0.], dtype=np.float64)
        pi2 = 2 * np.pi
        return pi2 * ratio.numerator * np.arange(0, 1, 1.0/ratio.denominator, dtype=np.float64)

    @property
    def sample(self):
        return self.angles(self.ratio)
        
    # Internal stuff
    
    @staticmethod
    def to_ratio(freq, limit=Sampler.rate):
        return Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(limit)

    @staticmethod
    def antialias(ratio):
        if Sampler.prevent_aliasing and abs(ratio) > Fraction(1,2):
            return Fraction(0, 1)
        if Sampler.prevent_aliasing and not Sampler.negative_frequencies and ratio < 0:
            return Fraction(0, 1)
        return ratio
    
    @staticmethod
    def wrap(ratio):
        # wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.
        n = ratio.numerator % ratio.denominator
        return Fraction(n, ratio.denominator)

    ### Representation ###

    def __repr__(self):
        return "Frequency(%s)" % self._hz

    def __str__(self):
        return "<Frequency: %s hz>" % self.hz

    ### Arithmetic ###
    
    def __nonzero__(self):
        """Zero frequency should be considered False"""
        return self.ratio != 0

    def __float__(self):
        return float(self.hz)

    def __int__(self):
        return int(self.hz)

    def __trunc__(self):
        """Returns an integral rounded towards zero."""
        return float(self._hz).__trunc__()

    def _op(op):
        """
        Add operator fallbacks. Usage: __add__, __radd__ = _gen_ops(operator.add)
        
        This function is borrowed and modified from fractions.Fraction._operator_fallbacks(),
        which generates forward and backward operator functions automagically.
        """
        def forward(a, b):
            if isinstance(b, a.__class__):
                return Frequency( op(a._hz, b._hz) )
            elif isinstance(b, (float, np.floating)):
                return Frequency( op(a._hz, b) )
            elif isinstance(b, Number):
                return Frequency( op(a._hz, b) )
            else:
                return NotImplemented
        forward.__name__ = '__' + op.__name__ + '__'
        forward.__doc__ = op.__doc__

        def reverse(b, a):
            if isinstance(a, Frequency):
                return Frequency( op(a._hz, b._hz) )
            elif isinstance(a, (float, np.floating)):
                return Frequency( op(a, b._hz) )
            elif isinstance(a, Number):
                return Frequency( op(a, b._hz) )
            else:
                return NotImplemented
        reverse.__name__ = '__r' + op.__name__ + '__'
        reverse.__doc__ = op.__doc__

        return forward, reverse

    __add__, __radd__ = _op(operator.add)
    __sub__, __rsub__ = _op(operator.sub)
    __mul__, __rmul__ = _op(operator.mul)
    __div__, __rdiv__ = _op(operator.div)
    __truediv__, __rtruediv__ = _op(operator.truediv)
    __floordiv__, __rfloordiv__ = _op(operator.div)

    __mod__, __rmod__ = _op(operator.mod)
    __pow__, __rpow__ = _op(operator.pow)

    def __pos__(self):
        return Frequency(self._hz)

    def __neg__(self):
        return Frequency(-self._hz)

    def __abs__(self):
        return Frequency(abs(self._hz))

    def __hash__(self):
        """hash(self), takes into account any rounding done on Frequency's initialisation."""
        return hash(self.hz)

    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            return cmp(self.ratio, other.ratio)
        else:
            return cmp(other, self.ratio)

    def __eq__(a, b):
        """a == b, takes into account any rounding done on Frequency's initialisation."""
        return a.hz == Frequency(b).hz

    def __lt__(a, b):
        return operator.lt(a._hz, b)

    def __gt__(a, b):
        return operator.gt(a._hz, b)

    def __le__(a, b):
        return operator.le(a._hz, b)

    def __ge__(a, b):
        return operator.ge(a._hz, b)

    # __reduce__ # TODO: Implement pickling -- see http://docs.python.org/library/pickle.html#the-pickle-protocol
    # __copy__, __deepcopy__

