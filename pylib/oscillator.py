#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi, exp
import numpy as np
import types
from numbers import Number


def to_phasor(x):
    return (abs(x), (phase(x) / (2 * pi) * 360))


class Sampler:
    rate = 44100


class Osc:
    """Oscillator class"""
    
    # Settings
    prevent_aliasing = True
    negative_frequencies = False
    
    # Root cache is a dictionary
    roots = dict()
    
    # This could be __new__ if Osc would extend Fraction or be immutable
    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self.ratio = Osc.limit_ratio(Fraction(*args))
        
        if not Osc.roots.has_key(self.period):
            Osc.roots[self.period] = self.table_roots() #self.gen_roots(self.nth_root)
        # self.roots = Osc.roots[self.period][self._root_order()]
        
    @classmethod
    def freq(cls, freq):
        """Oscillator can be initialized using a frequency, which can be a float."""
        ratio = Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(Sampler.rate)
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
    
    @property
    def ratio(self):
        return self.ratio
        
    @property
    def period(self):
        return self.ratio.denominator
        
    @property
    def order(self):
        return self.ratio.numerator
        
    @property
    def frequency(self):
        return float(self.ratio * Sampler.rate)
    
    ### Generating functions ###
    
    def normalize_fraction(self, n):
        n %= self.period
        return Fraction(n, self.period)
        
    def nth_root(self, n):
        w = 2 * pi
        return rect(1, w * self.normalize_fraction(n))
        
    def func_root(self, n):
        wi = 2 * pi * 1j
        return exp(wi * self.normalize_fraction(n))
        
    def gen_roots(self, func):
        return np.array(
            map( func, range(0, self.period) ),
            dtype=complex
        )
        
    def table_roots(self):
        if not Osc.roots.has_key(Sampler.rate):
            Osc.roots[Sampler.rate] = self.gen_table_roots()
        return Osc.roots[Sampler.rate][0:self.period] ** (Sampler.rate / float(self.period))
        
    def gen_table_roots(self):
        w = 2 * pi
        nth_root = lambda n: rect(1, w * Fraction(n, Sampler.rate))
        return np.array(map(nth_root, range(0, Sampler.rate)))
    
    ### Sampling ###
    
    def _root_order(self):
        return np.array(range(self.period)) * self.order % self.period
        
    @property
    def samples(self):
        if self.order == 1:
            return Osc.roots[self.period]
        else:
            return Osc.roots[self.period][self._root_order()]
        
    def __len__(self):
        return self.period
        
    def __getitem__(self, item):
        """Slicing support. If given a slice the behaviour will be:
        
        Stepping is wrapped modulo period.
        Negative and missing start value defaults to 0.
        Stop value is multiplied with wrapped step to give expected number of samples.
        """
        if isinstance(item, slice):
            # Step defaults to 1, is wrapped mod period, and can't be zero!
            step = ((item.step or 1) % self.period or 1)
            # None or negative start is 0
            start = max(item.start, 0)
            # Default to period
            stop = (item.stop or self.period) * step
            sl = slice(start, stop, step)
            indices = np.arange(*(sl.indices(sl.stop)))
            print indices[-1] % self.period # Could be used as cursor
            return self.samples[indices % self.period]
        else:
            return self.samples[np.array(item) % self.period]
    
    ### Representation ###
    
    def __eq__(self, other):
        return self.ratio == other.ratio
        
    def __repr__(self):
        return "Osc(%s, %s)" % (self.order, self.period)
        
    def __str__(self):
        return "<Osc: %s Hz>" % self.frequency
        
    def to_phasors(self):
        return np.array(map(to_phasor, self.samples))
        
    ### Arithmetic ###
    # 
    # See fractions.Fraction._operator_fallbacks() for automagic
    # generation of forward and backward operator functions
    def __add__(self, other):
        if isinstance(other, Osc):
            return Osc(self.ratio + other.ratio)
        elif isinstance(other, float):
            return Osc.freq(self.frequency + other)
        elif isinstance(other, Number):
            return Osc(self.ratio + other)
        else:
            return NotImplemented
    __radd__ = __add__
    
    def __mul__(self, other):
        if isinstance(other, Osc):
            return Osc(self.ratio * other.ratio)
        elif isinstance(other, float):
            return Osc.freq(self.frequency * other)
        elif isinstance(other, Number):
            return Osc(self.ratio * other)
        else:
            return NotImplemented
    __rmul__ = __mul__
    # def __rmul__(self, other):
    #     return self.__mul__(other)
        
if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.roots
    print o.to_phasors()
    print Osc.roots.keys()