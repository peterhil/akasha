#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from enthought.traits.api import HasTraits, DelegatesTo, PrototypedFrom, CFloat, Instance
from fractions import Fraction

from akasha.timing import sampler


class sampler(HasTraits):
    rate = Frequency( 44100 )


class Frequency(HasTraits):
    freq = CFloat()
    ratio = Instance( Fraction, depends_on = 'freq' )

    def __init__(self, value):
        self.freq = value
        self.ratio = self._get_ratio()
        # self.freq.on_trait_change(self._get_ratio())

    def _get_ratio(self):
        self.ratio = Fraction.from_float(float(self.freq)/sampler.rate).limit_denominator(sampler.rate)

    def __repr__(self):
        return "%.02f Hz" % self.freq


class Osc(HasTraits):
    """Oscillator class with Traits"""
    freq = Instance( Frequency )

    def __init__(self, freq):
        self.freq = freq

