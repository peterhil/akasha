#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi, exp
import numpy as np
import types

def to_phasor(x):
    return (abs(x), (phase(x) / (2 * pi) * 360))

def limit_ratio(f):
    if Osc.prevent_aliasing and abs(f) > Fraction(1,2):
        return Fraction(0, 1)
    if Osc.prevent_aliasing and not Osc.negative_frequencies and f < 0:
        return Fraction(0, 1)
    # wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.
    n = f.numerator % f.denominator
    return Fraction(n, f.denominator)

class Sampler:
    rate = 44100

class Osc:
    """Oscillator class"""
    
    # Settings
    prevent_aliasing = True
    negative_frequencies = False
    
    # Root cache is a dictionary
    roots = dict()
    
    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self.ratio = limit_ratio(Fraction(*args))
        
        if not Osc.roots.has_key(self.period):
            Osc.roots[self.period] = self.table_roots() # self.gen_roots(self.nth_root)
        self.roots = Osc.roots[self.period]
        self.roots = self.ord_roots()
    
    @classmethod
    def freq(self, freq):
        """Oscillator can be initialized using a frequency, which can be float."""
        ratio = Fraction.from_float(float(freq)/Sampler.rate).limit_denominator(Sampler.rate)
        return Osc(ratio)
    
    def _order_index(self, ratio):
        order, period = ratio.numerator, ratio.denominator
        return np.array(range(period)) * order % period
        
    def ord_roots(self):
        if self.order == 1:
            return self.roots
        else:
            return self.roots[self._order_index(self.ratio)]
    
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
    
    @property
    def frequency(self):
        return float(self.ratio * Sampler.rate)
    
    @property
    def ratio(self):
        return self.ratio
    
    @property
    def period(self):
        return self.ratio.denominator
    
    @property
    def order(self):
        return self.ratio.numerator
    
    def __eq__(self, other):
        return self.ratio == other.ratio
    
    def __repr__(self):
        return "Osc(%s, %s)" % (self.order, self.period)
    
    def to_phasors(self):
        return np.array(map(to_phasor, self.roots))
    
if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.roots
    print o.to_phasors()
    print Osc.roots.keys()