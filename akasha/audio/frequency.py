#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import operator

from fractions import Fraction
from numbers import Number

from .generators import PeriodicGenerator

from ..timing import sampler
from ..utils.decorators import memoized


class FrequencyRatioMixin:
    @property
    def frequency(self):
        return self._hz

    @frequency.setter
    def frequency(self, hz):
        self._hz = hz if isinstance(hz, Frequency) else Frequency(hz)  # Use Trellis or other Cells clone?

    @property
    def ratio(self):
        return self._hz.ratio

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator

    def __nonzero__(self):
        """Zero frequency should be considered False"""
        return self.ratio != 0

    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            return cmp(self.ratio, other.ratio)
        else:
            return cmp(other, self.ratio)

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * sampler.rate)

    def __float__(self):
        return float(self.hz)

    def __int__(self):
        return int(self.hz)


class Frequency(object, FrequencyRatioMixin, PeriodicGenerator):
    """Frequency class"""

    def __init__(self, hz, unwrapped=False):
        # Original frequency, independent of sampling rate or optimizations
        self._hz = float(hz)
        self.sampling = unwrapped

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * sampler.rate)

    @property
    def ratio(self):
        if not self.sampling:
            return self.wrap(self.antialias(self.to_ratio(self._hz)))
        else:
            return self.to_ratio(self._hz)

    @staticmethod
    @memoized
    def angles(ratio):
        """Normalized frequency (Tau) angles for one full period at ratio."""
        # Fastest generating method so far. Could be made still faster by using conjugate for half of the samples.
        if ratio == 0:
            return np.array([0.], dtype=np.float64)
        return ratio.numerator * np.arange(0, 1, 1.0/ratio.denominator, dtype=np.float64)

    @staticmethod
    def rads(ratio):
        """Radian angles for one full period at ratio."""
        return 2 * np.pi * Frequency.angles(ratio)

    @property
    def sample(self):
        return self.angles(self.ratio)
        
    # Internal stuff
    
    @staticmethod
    @memoized
    def to_ratio(freq, limit=sampler.rate):
        # @TODO investigate what is the right limit, and take beating tones into account!
        return Fraction.from_float(float(freq)/sampler.rate).limit_denominator(limit)

    @staticmethod
    def antialias(ratio):
        if sampler.prevent_aliasing and abs(ratio) > Fraction(1,2):
            return Fraction(0, 1)
        if sampler.prevent_aliasing and not sampler.negative_frequencies and ratio < 0:
            return Fraction(0, 1)
        return ratio
    
    @staticmethod
    def wrap(ratio):
        # wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.
        return ratio % 1

    ### Representation ###

    def __repr__(self):
        return "Frequency(%s)" % self._hz

    def __str__(self):
        return "<Frequency: %s hz>" % self.hz

    ### Arithmetic ###
    
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

