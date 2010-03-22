#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi
import numpy as np
from recipes import prop

class SampleRate(object):
    """
    A descriptor object for sample rate.
    TODO: also handle sending resampling calls to objects?
    """
    default_rate = 44100
    
    def __init__(self, value=default_rate):
        self.rate = value
    
    def __get__(self, obj, objtype):
        print "Self: %s, Obj: %s, Object type: %s" % (self.rate, obj, objtype)
        return self.rate or __set__(default_rate)
    
    def __set__(self, obj, rate):
        print "set"
        self.rate = rate

class Osc:
    """Oscillator class"""
    roots = dict()
    tuning = SampleRate()
    
    def __init__(self, *args):
        # Set ratio and limit between 0/1 and 1/1
        self.ratio = Fraction(*args) % Fraction(1)
        
        root = lambda x: rect(1, 2 * pi * Fraction(x, self.period))
        if not Osc.roots.has_key(self.period):
            Osc.roots[self.period] = np.array(map(root, range(0, self.period)), dtype=complex)
        self.roots = Osc.roots[self.period]
    
    @property
    def ratio(self):
        return self.ratio
        
    @property
    def period(self):
        return self.ratio.denominator
        
    @property
    def order(self):
        return self.ratio.numerator % self.period
        
    def __eq__(self, other):
        return self.ratio == other.ratio
        
    def __repr__(self):
        return "Osc(%s)" % self.ratio.__repr__()
        
if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.roots
    print map(lambda x: (abs(x), (phase(x) / (2 * pi) * 360)), o.roots)
    print Osc.roots.keys()
    # print Osc(Fraction('1/2')).roots