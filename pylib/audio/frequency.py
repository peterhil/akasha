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


class Hz(object, PeriodicGenerator):
    def __init__(self, hz=0):
        self.__hz = hz

    def __get__(self, instance, owner):
        print "Getting from Hz: ", self, instance, owner
        return self.hz

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * Sampler.rate)

    @property
    def ratio(self):
        return self.to_ratio(self.__hz)

    @staticmethod
    def to_ratio(freq, limit=Sampler.rate):
        return Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(limit)

class Game(object):
    f0 = Hz()
    f10 = Hz(hz=10)
    f20 = Hz(hz=20)
    
    def __init__(self, hz=100):
        self.__class__.foo = Hz(hz)

class Frequency(object, PeriodicGenerator):
    """Frequency class"""

    def __init__(self, hz):
        # Original frequency, independent of sampling rate or optimizations
        self.__hz = float(hz)

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * Sampler.rate)

    def __get__(self, instance, owner):
        print "Getting from Frequency: ", self, instance, owner
        return self

    def __nonzero__(self):
        """Zero frequency should be considered False"""
        return self.ratio != 0

    @property
    def ratio(self):
        return self.wrap(self.antialias(self.to_ratio(self.__hz)))

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * Sampler.rate)
    
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

    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            return cmp(self.ratio, other.ratio)
        else:
            return cmp(other, self.ratio)

    def __repr__(self):
        return "Frequency(%s)" % self.__hz

    def __str__(self):
        return "<Frequency: %s hz>" % self.hz

    ### Arithmetic ###
    
    def __float__(self):
        return float(self.hz)

    def __int__(self):
        return int(self.hz)

    def _op(op):
        """
        Add operator fallbacks. Usage: __add__, __radd__ = _gen_ops(operator.add)
        
        This function is borrowed and modified from fractions.Fraction._operator_fallbacks(),
        which generates forward and backward operator functions automagically.
        """
        def forward(a, b):
            if isinstance(b, a.__class__):
                return Frequency( op(a.__hz, b.__hz) )
            elif isinstance(b, (float, np.floating)):
                return Frequency( op(a.__hz, b) )
            elif isinstance(b, Number):
                return Frequency( op(a.__hz, b) )
            else:
                return NotImplemented
        forward.__name__ = '__' + op.__name__ + '__'
        forward.__doc__ = op.__doc__

        def reverse(b, a):
            if isinstance(a, Frequency):
                print "Doing reverse with frequency"
                return Frequency( op(a.__hz, b.__hz) )
            elif isinstance(b, (float, np.floating)):
                return Frequency( op(a.__hz, b) )
            elif isinstance(b, Number):
                return Frequency( op(a.__hz, b) )
            else:
                return NotImplemented
        reverse.__name__ = '__' + op.__name__ + '__'
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

    # Also implemented by Fractions:
    
    # __pos__, __neg__, __abs__
    # __trunc__
    # __hash__, __eq__
    # __lt__, __gt__, __le__, __ge__
    # __reduce__, __copy__, __deepcopy__

