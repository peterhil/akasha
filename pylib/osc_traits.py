#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
# from cmath import rect, polar, phase, pi, exp
from fractions import Fraction

from enthought.traits.api import HasTraits, \
    DelegatesTo, PrototypedFrom, Float, CFloat, Int, Instance, Property, Str

# My modules
# from generators import PeriodicGenerator
# from timing import Sampler



class Sampler(HasTraits):
    rate = Frequency( 44100 )


class Frequency(HasTraits):
    freq = CFloat()
    ratio = Instance( Fraction, depends_on = 'freq' )
    
    def __init__(self, value):
        self.freq = value
        self.ratio = self._get_ratio()
        # self.freq.on_trait_change(self._get_ratio())
    
    def _get_ratio(self):
        self.ratio = Fraction.from_float(float(self.freq)/Sampler.rate).limit_denominator(Sampler.rate)
    
    def __repr__(self):
        return "%.02f Hz" % self.freq


class Osc(HasTraits):
    """Oscillator class with Traits"""
    freq = Instance( Frequency )
    
    def __init__(self, freq):
        self.freq = freq
    