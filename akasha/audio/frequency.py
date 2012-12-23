#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import operator

from fractions import Fraction
from numbers import Number

from akasha.audio.generators import PeriodicGenerator
from akasha.timing import sampler
from akasha.types.numeric import RealUnit
from akasha.utils import _super
from akasha.utils.decorators import memoized


class FrequencyRatioMixin(object):
    @classmethod
    def from_ratio(cls, ratio, den=False, *args, **kwargs):
        if den:
            ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * sampler.rate, *args, **kwargs)

    @property
    def frequency(self):
        return self._hz

    @frequency.setter
    def frequency(self, hz):
        # Use Trellis or other Cells clone?
        self._hz = float(hz) if isinstance(self, Frequency) else Frequency(hz)

    @property
    def ratio(self):
        return self.wrap(self.antialias(self.to_ratio(self._hz)))

    @property
    def hz(self):
        "Depends on original frequency's rational approximation and sampling rate."
        return float(self.ratio * sampler.rate)

    @property
    def period(self):
        return self.ratio.denominator

    @property
    def order(self):
        return self.ratio.numerator

    @staticmethod
    @memoized
    def to_ratio(freq, limit=sampler.rate):
        # @TODO investigate what is the right limit, and take beating tones into account!
        return Fraction.from_float(float(freq) / sampler.rate).limit_denominator(limit)

    @staticmethod
    def antialias(ratio):
        if sampler.prevent_aliasing and abs(ratio) > Fraction(1, 2):
            return Fraction(0, 1)
        if sampler.prevent_aliasing and not sampler.negative_frequencies and ratio < 0:
            return Fraction(0, 1)
        return ratio

    @staticmethod
    def wrap(ratio):
        # wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.
        return ratio % 1

    def __nonzero__(self):
        """Zero frequency should be considered False"""
        return self.ratio != 0

    def _cmp(op):
        """Generate comparison methods."""

        def comparison(self, other):
            if isinstance(other, FrequencyRatioMixin):
                return op(self.ratio, other.ratio)
            elif isinstance(other, Number):
                return op(float(self), float(other))
            else:
                return NotImplemented

        comparison.__name__ = '__' + op.__name__ + '__'
        comparison.__doc__ = op.__doc__

        return comparison

    __eq__ = _cmp(operator.eq)
    __ge__ = _cmp(operator.ge)
    __gt__ = _cmp(operator.gt)
    __le__ = _cmp(operator.le)
    __lt__ = _cmp(operator.lt)

    def __float__(self):
        return float(self.hz)

    def __int__(self):
        return int(self.hz)


class Frequency(FrequencyRatioMixin, RealUnit, PeriodicGenerator):
    """Frequency class"""

    def __init__(self, hz, unwrapped=False):
        self._unit = '_hz'
        self._hz = float(hz) # Original frequency, independent of sampling rate or optimizations
        self.unwrapped = unwrapped

    @property
    def ratio(self):
        if self.unwrapped:
            return self.to_ratio(self._hz)
        return _super(self).ratio

    @staticmethod
    @memoized
    def angles(ratio):
        """Normalized frequency (Tau) angles for one full period at ratio."""
        # Could be made still faster by using conjugate for half of the samples.
        zero = np.zeros(1, dtype=np.float64)
        if np.all(zero == ratio):
            return zero
        return ratio.numerator * np.arange(0, 1, 1.0 / ratio.denominator, dtype=np.float64)

    @staticmethod
    def rads(ratio):
        """Radian angles for one full period at ratio."""
        return 2 * np.pi * Frequency.angles(ratio)

    @property
    def sample(self):
        return self.angles(self.ratio)

    def __repr__(self):
        return "Frequency(%s)" % self._hz

    def __str__(self):
        return "<Frequency: %s hz>" % self.hz

    def __hash__(self):
        """hash(self), takes into account any rounding done on Frequency's initialisation."""
        return hash(self.ratio)

    def __eq__(self, other):
        """a == b, takes into account any rounding done on Frequency's initialisation."""
        return self.ratio == Frequency(other).ratio

    # TODO: Implement pickling
    # http://docs.python.org/library/pickle.html#the-pickle-protocol

    # __reduce__
    # __copy__, __deepcopy__

