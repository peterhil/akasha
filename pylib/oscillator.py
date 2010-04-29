#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi, exp
import numpy as np
import types
from numbers import Number
from timing import Sampler

# np.set_printoptions(precision=4, suppress=True)

def to_phasor(x):
    return (abs(x), (phase(x) / (2 * pi) * 360))


class Osc:
    """Oscillator class
    
    Viewing complex samples as pairs of reals:
    
    o = Osc(1,8)
    a = o.samples.copy()
    a
    b = a.view(np.float64).reshape(8,2)
    b *= np.array([2,0.5])
    b
    c = b.view(np.complex128)
    c
    b.transpose()
    b.transpose()[0]
    """
    
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
            # Osc.roots[self.period] = self.gen_roots(self.func_root)   # 0.927 s
            # Osc.roots[self.period] = self.gen_roots(self.nth_root)    # 0.731 s
            # Osc.roots[self.period] = self.table_roots() # 0.057 s
            Osc.roots[self.period] = self.np_exp() # 0.042 s
        # self.roots = Osc.roots[self.period][self._root_order()]
        
    @classmethod
    def freq(cls, freq):
        """Oscillator can be initialized using a frequency. Can be a float."""
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
        
    def np_exp(self):
        """Fastest generating method so far. Uses numpy.exp with linspace for angles.
        Could be made still faster by using conjugate for half of the samples."""
        return np.exp(np.linspace(0, 2 * pi, self.period + 1) * 1j)[:-1]
    
    ### Sampling ###
    
    def _root_order(self):
        return np.arange(self.period) * self.order % self.period
        
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
        
        # Step defaults to 1, is wrapped modulo period, and can't be zero!
        # Start defaults to 0, is wrapped modulo period
        # Number of elements returned is the absolute differerence of 
        # stop - start (or period and 0 if either value is missing)
        # Element count is multiplied with step to produce the same 
        # number of elements for different step values.
        """
        if isinstance(item, slice):
            step = ((item.step or 1) % self.period or 1)
            start = ((item.start or 0) % self.period)
            element_count = abs((item.stop or self.period) - (item.start or 0))
            stop = start + (element_count * step)
            # Construct an array of indices.
            item = np.arange(*(slice(start, stop, step).indices(stop)))
            # print item[-1] % self.period # Could be used as cursor
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


if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.roots
    print o.to_phasors()
    print Osc.roots.keys()