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
        if owner == Sampler:
            return self.__hz
        else:
            return float(self.ratio * Sampler.rate) # FIXME

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

    def _op(op):
        """
        Add operator fallbacks. Usage: __add__, __radd__ = _gen_ops(operator.add)
        
        This function is borrowed and modified from fractions.Fraction._operator_fallbacks(),
        which generates forward and backward operator functions automagically.
        """
        def forward(a, b):
            if isinstance(b, a.__class__):
                return Frequency( op(a.__hz, b.__hz) )
            elif isinstance(b, float):
                return Frequency( op(a.__hz, b) )
            elif isinstance(b, Number):
                return Frequency( op(a.__hz, b) )
            else:
                return NotImplemented
        forward.__name__ = '__' + op.__name__ + '__'
        forward.__doc__ = op.__doc__

        def reverse(b, a):
            if isinstance(a, Frequency):
                return Frequency( op(a.__hz, b.__hz) )
            elif isinstance(b, float):
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


class Osc(object, PeriodicGenerator):
    """Oscillator class"""

    def __init__(self, freq):
        self._frequency = Frequency(freq)
        self.superness = (2,2,2,2)  # CLEANUP: Oscs shouldn't know about superness -> move curves to own class!
        self.roots = self.np_exp
        self.roots(self.ratio)

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * Sampler.rate)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, hz):
        if isinstance(hz, Frequency):
            self._frequency = hz
        else:
            self._frequency = Frequency(hz)  # Use Trellis, and make a interface for frequencies

    @property
    def ratio(self):
        return self._frequency.ratio

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator

    ### Generating functions ###

    @staticmethod
    @memoized
    def np_exp(ratio):
        """Fastest generating method so far. Uses numpy.exp with linspace for angles.
        Could be made still faster by using conjugate for half of the samples."""
        if ratio == 0:
            return np.exp(np.array([0j]))
        pi2 = 2 * np.pi
        # return np.exp(1j * np.linspace(0, pi2, ratio.denominator, endpoint=False))
        return np.exp(ratio.numerator * 1j * pi2 * np.arange(0, 1, 1.0/ratio.denominator))  # 53.3 us per loop

    @staticmethod
    @memoized
    def angles(ratio):
        if ratio == 0:
            return np.array([0.], dtype=np.float64)
        pi2 = 2 * np.pi
        return pi2 * ratio.numerator * np.arange(0, 1, 1.0/ratio.denominator, dtype=np.float64)
    
    @staticmethod
    @memoized
    def circle(ratio):
        return np.exp(1j * Osc.angles(ratio))

    # Older alternative (and sometimes more precise) ways to generate roots

    @staticmethod
    @memoized
    def func_roots(ratio):
        # self.roots = self.func_roots        # 0.108 s
        wi = 2 * pi * 1j
        return np.exp(wi * ratio * np.arange(0, ratio.denominator))

    def table_roots(self, ratio):
        # self.roots = self.table_roots       # 0.057 s
        roots = self.np_exp(Fraction(1, Sampler.rate))
        return roots[0:ratio.denominator] ** (Sampler.rate / float(ratio.denominator))


    ### Sampling ###

    @property
    def sample(self):
        return self.np_exp(self.ratio)

    @property
    def imag(self):
        return self.sample.imag

    ### Representation ###

    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            return cmp(self.ratio, other.ratio)
        else:
            return cmp(other, self.ratio)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.frequency)

    def __str__(self):
        return "<%s: %s hz>" % (self.__class__.__name__, self.frequency)



class Super(Osc):
    """Oscillator that has a superness parameter."""
    
    def __init__(self, freq, m=4.0, n=2.0, p=2.0, q=2.0, a=1.0, b=1.0):
        """
        Super oscillator can be initialized using a frequency and superness.
        
        See 'Superellipse' article at Wikipedia for explanation of this parameter means:
        http://en.wikipedia.org/wiki/Superellipse
        """
        self._frequency = Frequency(freq)
        self.amp = 1.0
        self.superness = (m, n, p, q, a, b)
        self.gen(self.ratio, self.superness)

    @classmethod
    def freq(cls, freq, *superness):
        return cls(freq, superness)

    @classmethod
    def from_ratio(cls, ratio, den=False, *superness):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * Sampler.rate, superness)

    @staticmethod
    def normalise_superness(superness):
        if superness == None:
            logger.warn("Got None for superness!")
            return (2,2,2,2) # identity for superness
        if isinstance(superness, tuple) and len(superness) == 4:
            return superness
        if not isinstance(superness, (list, tuple, Number)):
            raise ValueError("Superness %s needs to be a number, a tuple or a list of length one to four. " + \
                "Got type %s" % (superness, type(superness)))
        if isinstance(superness, Number):
            superness = tuple([superness])
        if len(superness) < 4:
            superness = list(superness) + [superness[-1]] * (4 - len(superness))
        return tuple(superness[:4])  # Take only the first four params

    ### Generating functions ###

    @staticmethod
    @memoized
    def gen(ratio, superness):
        angles = Super.angles(ratio)
        return Super.superformula(angles, superness) * np.exp(1j * angles)

    @staticmethod
    def superformula(angles, superness):
        """Superformula function. Generates amplitude curves applicable to oscillators by multiplying.

        Usage:
        supercurve(angles, superness)
        s = Super(431, m, n, p, q, a, b), where m is number of spikes and n-q determine the roundness.

        For more information, see:
        http://en.wikipedia.org/wiki/Superellipse and
        http://en.wikipedia.org/wiki/Superformula
        """
        (m, n, p, q, a, b) = list(superness)
        assert np.isscalar(m), "%s in superformula is not scalar." % m
        coeff = angles * (m / 4.0)
        return (np.abs(np.cos(coeff) / a)**p + np.abs(np.sin(coeff) / b)**q) ** (-1.0/n)

    ### Sampling ###

    @property
    def sample(self):
        #return normalize(self.superformula(self.np_exp(self.ratio), self.superness))
        return normalize(self.gen(self.ratio, self.superness)) * self.amp

    def __repr__(self):
        return "%s(%s, superness=%s)" % (self.__class__.__name__, self.frequency, self.superness)

    def __str__(self):
        return "<%s: %s hz, superness %s>" % (self.__class__.__name__, self.frequency, self.superness)

