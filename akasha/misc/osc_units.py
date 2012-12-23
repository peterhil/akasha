#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import quantities as pq
import types

from cmath import rect, pi, exp
from fractions import Fraction
from numbers import Number

from akasha.audio.generators import PeriodicGenerator
from akasha.control.io.audio import play, write
from akasha.timing import sampler
from akasha.utils.decorators import memoized
from akasha.utils.math import to_phasor


pq.markup.config.use_unicode = True  # Use unicode units representation

### Units
Qt = pq.Quantity
Hz = pq.Hz


class Unitsampler(object):

    _rate = Qt(44100.0, Hz)

    @property
    def rate(self):
        return self._rate

    def _set_frame_time(self):
        pq.frame = pq.UnitTime('frame', sampler.rate**-1, symbol='frame')

    @rate.setter
    def rate(self, value):
        self._rate = value
        self._set_frame_time()


class Osc(object, PeriodicGenerator):
    """Oscillator class"""

    # Settings
    prevent_aliasing = True
    negative_frequencies = False

    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self._ratio = Osc.limit_ratio(Fraction(*args))

        # Generate roots
        # self.roots = self.func_roots        # 0.108 s
        # self.roots = self.table_roots       # 0.057 s
        self.roots = self.np_exp
        self.roots(self.ratio)

    @classmethod
    def freq(cls, freq):
        """Oscillator can be initialized using a frequency. Can be a float."""
        ratio = Fraction.from_float(float(freq)/sampler.rate).limit_denominator(sampler.rate)
        return cls(ratio)

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
        return float(self.ratio * sampler.rate)

    ### Generating functions ###

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

    @memoized
    def func_roots(ratio):
        wi = 2 * pi * 1j
        return np.exp(wi * ratio * np.arange(0, ratio.denominator))

    def table_roots(self, ratio):
        roots = self.np_exp(Fraction(1, sampler.rate))
        return roots[0:ratio.denominator] ** (sampler.rate / float(ratio.denominator))

    ### Sampling ###

    @property
    def sample(self):
        return self.np_exp(self.ratio)

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
            return Osc(self.ratio * other.ratio)
        elif isinstance(other, float):
            return Osc.freq(self.frequency * other)
        elif isinstance(other, Number):
            return Osc(self.ratio * other)
        elif isinstance(other, np.ndarray) and isinstance(other[0], np.number):
            return np.array(map(Osc.freq, self.frequency * other), dtype=np.object)
        else:
            raise NotImplementedError("Self: %s, other: %s", (self, other))
    __rmul__ = __mul__


if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.np_exp(o.period)
    print to_phasor(o.sample)

