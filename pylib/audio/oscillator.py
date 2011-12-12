#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np
from cmath import rect, pi, exp, phase
from fractions import Fraction

# Types
import types
from numbers import Number

# My modules
from audio.generators import PeriodicGenerator
from timing import Sampler

# Utils
from utils.decorators import memoized
from utils.math import *

# Settings
np.set_printoptions(precision=16, suppress=True)


class Hz(object, PeriodicGenerator):
    def __init__(self, hz=0):
        self.__frequency = hz

    def __get__(self, instance, owner):
        print "Getting from Hz: ", self, instance, owner
        return self.hz

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * Sampler.rate)

    @property
    def ratio(self):
        return self.to_ratio(self.__frequency)

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
        self._frequency = hz

    def __get__(self, instance, owner):
        print "Getting from Frequency: ", self, instance, owner
        return float(self.ratio * Sampler.rate)

    def __nonzero__(self):
        """Zero frequency should be considered False"""
        return self.ratio != 0

    @property
    def ratio(self):
        return self.wrap(self.antialias(self.to_ratio(self._frequency)))

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
        return "Frequency(%s)" % self._frequency

    def __str__(self):
        return "<Frequency: %s hz>" % self.hz

    ### Arithmetic ###
    #
    # See fractions.Fraction._operator_fallbacks() for automagic
    # generation of forward and backward operator functions
    
    def __float__(self):
        return float(self.hz)
    
    def __add__(self, other):
        print "Self: %s, other: %s (%s)" % (self, other, type(other))

        if isinstance(other, self.__class__):
            return Frequency(self.hz + other.hz)
        elif isinstance(other, float):
            return Frequency(self.hz + other)
        elif isinstance(other, complex):
            return complex(self.hz + other)
        elif isinstance(other, Number):
            return Frequency(float(self.ratio) + other)
        elif type(other) in np.typeDict.values():
            return np.typeDict[np.typeNA[type(other)]](self.ratio * other)
        elif isinstance(other, np.ndarray) and isinstance(other[0], np.number):
            return np.array(map(Frequency, self.hz + other), dtype=np.object)
        else:
            raise NotImplementedError("Self: %s, other: %s", (self, other))
    __radd__ = __add__

    def __mul__(self, other):
        print "Self: %s, other: %s (%s)" % (self, other, type(other))
        
        if isinstance(other, self.__class__):
            return Frequency(self.hz * other.hz)
        elif isinstance(other, float):
            return Frequency(self.hz * other)
        elif isinstance(other, complex):
            return complex(self.hz * other)
        elif isinstance(other, Number):
            return Frequency(float(self.ratio) * other)
        elif type(other) in np.typeDict.values():
            return np.typeDict[np.typeNA[type(other)]](self.ratio * other)
        elif isinstance(other, np.ndarray) and isinstance(other[0], np.number):
            return map(lambda f: Frequency(self.hz * f), other)
        else:
            raise NotImplementedError("Self: %s, other: %s", (self, other))
    __rmul__ = __mul__
    

class Osc(object, PeriodicGenerator):
    """Oscillator class"""

    def __init__(self, freq):
        self._frequency = Frequency(freq)
        self.roots = self.np_exp
        self.roots(self.ratio)

    @classmethod
    def freq(cls, freq):
        return cls(freq)

    @classmethod
    def from_ratio(cls, ratio, den=False):
        if den: ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * Sampler.rate)

    @property
    def hz(self):
        return self._frequency.hz

    @hz.setter
    def hz(self, hz):
        self._frequency._frequency = hz

    @property
    def frequency(self):
        return self._frequency.hz

    @frequency.setter
    def frequency(self, hz):
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
        pi2 = 2 * pi
        # return np.exp(1j * np.linspace(0, pi2, ratio.denominator, endpoint=False))
        return np.exp(ratio.numerator * 1j * pi2 * np.arange(0, 1, 1.0/ratio.denominator))  # 53.3 us per loop

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
        return "<%s: %s hz>" % (self.__class__.__name__, self.hz)



class Super(Osc):
    """Oscillator that has a superness parameter."""
    
    def __init__(self, ratio, superness=2):
        """Super (oscillator) can be initialized using a frequency and superness. See 'superellipse' at Wikipedia for explanation of this parameter means."""
        super(self.__class__, self).__init__(ratio)
        self.superness = self.normalise_superness(superness)

    @classmethod
    def freq(cls, freq, superness=2):
        return cls(freq, superness)

    @staticmethod
    def normalise_superness(superness):
        if not isinstance(superness, (list, Number)):
            raise ValueError("Superness needs to be a number or a list of length one to four.")
        if isinstance(superness, Number):
            superness = [superness]
        if len(superness) < 4:
            superness += [superness[-1]] * (4 - len(superness))
        return superness[:4]  # Take only the first four params

    def supercurve(self, points):
        """Superformula function. Generates amplitude curves applicable to oscillators by multiplying.

        Usage:
        supercurve(m, n1, n2, n3), where m is number of spikes and n1-3 determine the roundness.
        o = Osc.freq(431, superness=[3, 0.5])

        For more information, see:
        http://en.wikipedia.org/wiki/Superellipse and
        http://en.wikipedia.org/wiki/Superformula"""

        (m, n1, n2, n3) = self.superness
        angles = np.array(map(phase, points))
        super = (np.abs(np.cos(m * angles / 4.0))**n2 + np.abs(np.sin(m * angles / 4.0))**n3) ** -1.0/n1

        return points * super

    ### Sampling ###

    @property
    def sample(self):
        return normalize(self.supercurve(self.np_exp(self.ratio)))

    def __repr__(self):
        return "%s(%s, superness=%s)" % (self.__class__.__name__, self.frequency, self.superness)

    def __str__(self):
        return "<%s: %s hz, superness %s>" % (self.__class__.__name__, self.hz, self.superness)


if __name__ == '__main__':
    from utils.graphing import *
    o = Osc(Fraction(1, 8))
    t = slice(0, Sampler.rate)
    print o.np_exp(o.period)
    print to_phasors(o.sample)

