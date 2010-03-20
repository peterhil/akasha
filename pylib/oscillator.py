#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from fractions import Fraction
from cmath import rect, polar, phase, pi
import numpy as np

# def root(x):
#     cmath.polar(1, cmath.pi*x/period)

class Osc:
    """Oscillator class"""
    roots = dict()
    
    def __init__(self, ratio):
        self.order, self.period = ratio.numerator, ratio.denominator
        
        root = lambda x: rect(1, 2 * pi * Fraction(x, self.period))
        if not Osc.roots.has_key(ratio):
            Osc.roots[ratio] = np.array(map(root, range(0, self.period)), dtype=complex)
        self.roots = Osc.roots[ratio]
    
if __name__ == '__main__':
    o = Osc(Fraction(1, 8))
    print o.roots
    print map(lambda x: phase(x) / (2 * pi) * 360, o.roots)
    print Osc.roots.keys()
    # print Osc(Fraction('1/2')).roots