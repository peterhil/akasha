#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enthought.traits.api import HasTraits, CFloat, Instance
# from enthought.traits.api import DelegatesTo, PrototypedFrom
from fractions import Fraction


class Frequency(HasTraits):
    freq = CFloat()
    ratio = Instance(Fraction, depends_on='freq')

    def __init__(self, value):
        self.freq = value
        self.ratio = self._get_ratio()
        # self.freq.on_trait_change(self._get_ratio())

    def _get_ratio(self):
        self.ratio = Fraction.from_float(
            float(self.freq) / sampler.rate
        ).limit_denominator(sampler.rate)

    def __repr__(self):
        return "%.02f Hz" % self.freq


class sampler(HasTraits):
    rate = Frequency(44100)


class Osc(HasTraits):
    """Oscillator class with Traits"""
    freq = Instance(Frequency)

    def __init__(self, freq):
        self.freq = freq
