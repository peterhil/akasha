#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np
from cmath import rect, pi, exp, phase
from fractions import Fraction

# Types
import types
import quantities as pq
from numbers import Number

# My modules
from generators import PeriodicGenerator
from timing import Sampler

# Utils
from utils.decorators import memoized
from utils.math import *
from utils.graphing import *
from utils import play, write

# Settings
np.set_printoptions(precision=16, suppress=True)
pq.markup.config.use_unicode = True  # Use unicode units representation

class Acc(object, PeriodicGenerator):
    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self._ratio = Osc.limit_ratio(Fraction(*args))
        self.roots = self.gen
        self.roots(self._ratio)

    def gen(self, ratio):
        if ratio == 0:
            return np.exp(np.array([0j]))
        pi2 = 2 * np.pi
        return ratio.numerator * 1j * pi2 * np.arange(0, 1, 1.0/ratio.denominator)

    @classmethod
    def freq(cls, freq):
        ratio = Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(Sampler.rate)
        return cls(ratio)

    ### Properties ###

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator

    @property
    def frequency(self):
        return float(self.ratio * Sampler.rate)

    ### Sampling ###

    @property
    def sample(self):
        return self.gen(self.ratio)


class Osc(object, PeriodicGenerator):
    """Oscillator class"""

    # Settings
    prevent_aliasing = True
    negative_frequencies = False

    def __init__(self, ratio, superness=2):
        # print "Ratio: %s, Superness: %s" % (ratio, superness)
        # Set ratio and limit between 0/1 and 1/1
        self._ratio = Osc.limit_ratio(Fraction(*[ratio]))

        # Set superness
        if not isinstance(superness, (list, Number)):
            raise ValueError("Superness needs to be a number or a list of length one to four.")
        if isinstance(superness, Number):
            superness = [superness]
        if len(superness) < 4:
            superness += [superness[-1]] * (4 - len(superness))

        self.superness = superness[:4]  # Take only the first four params

        # Generate roots
        # self.roots = self.func_roots        # 0.108 s
        # self.roots = self.table_roots       # 0.057 s
        self.roots = self.np_exp
        self.roots(self.ratio)

    @staticmethod
    def limit(f, max=44100):
        return Fraction(int(round(f * max)), max)

    @classmethod
    def freq(cls, freq, superness=2, rounding='native'):
        """Oscillator can be initialized using a frequency. Can be a float."""
        ratio = Fraction.from_float(float(freq)/Sampler.rate)
        if rounding == 'native':
            ratio = ratio.limit_denominator(Sampler.rate)
        else:
            ratio = cls.limit(ratio, Sampler.rate)
        return cls(ratio, superness)

    @staticmethod
    def limit_ratio(f):
        if Osc.prevent_aliasing and abs(f) > Fraction(1,2):
            return Fraction(0, 1)
        if Osc.prevent_aliasing and not Osc.negative_frequencies and f < 0:
            return Fraction(0, 1)
        # wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.
        n = f.numerator % f.denominator
        return Fraction(n, f.denominator)

    ### Properties ###

    @property
    def ratio(self): return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value

    @property
    def period(self): return self.ratio.denominator

    @property
    def order(self): return self.ratio.numerator

    @property
    def frequency(self):
        return float(self.ratio * Sampler.rate)

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
        wi = 2 * pi * 1j
        return np.exp(wi * ratio * np.arange(0, ratio.denominator))

    def table_roots(self, ratio):
        roots = self.np_exp(Fraction(1, Sampler.rate))
        return roots[0:ratio.denominator] ** (Sampler.rate / float(ratio.denominator))

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
        if (self.superness == [2,2,2,2]):
            return self.np_exp(self.ratio)
        else:
            return normalize(self.supercurve(self.np_exp(self.ratio)))

    ### Representation ###

    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            return cmp(self.ratio, other.ratio)
        else:
            return cmp(other, self.ratio)

    def __repr__(self):
        return "Osc(%s, %s)" % (self.order, self.period)

    def __str__(self):
        return "<Osc: %s Hz>" % self.frequency

    ### Arithmetic ###
    #
    # See fractions.Fraction._operator_fallbacks() for automagic
    # generation of forward and backward operator functions
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Osc(self.ratio + other.ratio)
        elif isinstance(other, float):
            return Osc.freq(self.frequency + other)
        elif isinstance(other, Number):
            return Osc(self.ratio + other)
        elif isinstance(other, np.ndarray) and isinstance(other[0], np.number):
            return np.array(map(Osc.freq, self.frequency + other), dtype=np.object)
        else:
            raise NotImplementedError("Self: %s, other: %s", (self, other))
    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            superness = list(np.array(self.superness) * np.array(other.superness) / 2.0) # TODO Is this the right behaviour?
            return Osc(self.ratio * other.ratio, superness)
        elif isinstance(other, float):
            return Osc.freq(self.frequency * other, self.superness)
        elif isinstance(other, Number):
            return Osc(self.ratio * other, self.superness)
        elif isinstance(other, np.ndarray) and isinstance(other[0], np.number):
            return map(lambda f: Osc.freq(self.frequency * f, self.superness), other)
        else:
            raise NotImplementedError("Self: %s, other: %s", (self, other))
    __rmul__ = __mul__


if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    t = slice(0, Sampler.rate)
    print o.np_exp(o.period)
    print to_phasors(o.sample)
