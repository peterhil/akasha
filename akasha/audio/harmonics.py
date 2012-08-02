#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import exceptions
import numpy as np

from fractions import Fraction
from copy import copy, deepcopy

from .envelope import Exponential, Gamma
from .frequency import Frequency, FrequencyRatioMixin
from .generators import Generator
from .oscillator import Osc

from ..timing import sampler
from ..utils.decorators import memoized
from ..utils.log import logger
from ..utils.math import random_phase, normalize


class Overtones(object, FrequencyRatioMixin, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(self, sndobj=Osc(216.0), n=8, func=lambda x: 1+x, damping=None, rand_phase=False):
        self.base = sndobj
        # TODO Setting ovt.frequency (ovt._hz) leaves ovt.base.frequency (ovt.base._hz)
        # where it was -- is this the desired behaviour?
        self._hz = self.base.frequency
        self.n = n
        self.func = func
        # Sine waves FIXME: separate freq. damping from rate
        #self.damping = damping or (lambda f, a=1.0: (-5*np.log2(float(f))/(10.0), a*float(self.frequency)/float(f)))
        #self.damping = damping or (lambda f, a=1.0: -5*np.log2(float(f))/10.0)
        self.damping = damping or (lambda f, a=1.0: -5*np.log2(float(f))/1000.0)
        self.sustain = None
        self.sustained = None
        self.rand_phase = rand_phase

    @property
    def max_overtones(self):
        low_freq_overtone_limit = 10
        return int(sampler.rate / (2.0 * max(self.frequency, low_freq_overtone_limit)))

    @property
    def limit(self):
        return max(min(self.max_overtones, self.n), 1)

    @property
    def overtones(self):
        return np.apply_along_axis(self.func, 0, np.arange(0, self.limit, dtype=np.float32))

    @property
    def oscs(self):
        # TODO cleanup - make an interface for different Oscs!
        return self.gen_oscs()

    def gen_oscs(self):
        base = self.base.__class__
        if base.__name__ == 'Super':
            oscs = map(lambda f: base(f, *self.base.superness), float(self.frequency) * self.overtones)
        else:
            oscs = map(base, float(self.frequency) * self.overtones)
        return oscs

    def sample(self, iter):
        if isinstance(iter, int):
            frames = np.array([0j])
        else:
            frames = np.zeros(len(iter), dtype=complex)
        
        for o in self.oscs:
        # for f in self.overtones:
            # o = deepcopy(self.base)     # Uses deepcopy to preserve superness & other attrs
            # o.frequency = Frequency(self.frequency * f)
            #logger.boring("Overtone frequency: %s" % o.frequency)
            if o.frequency == 0: break
            
            # e = Exponential(0, amp=float(self.frequency/o.frequency*float(self.frequency))) # square waves
            # e = Exponential(0, amp=float(self.frequency**2/o.frequency**2*float(self.frequency))) # triangle waves
            # e = Exponential(-o.frequency/100.0) # sine waves
            e = Exponential(self.damping(o.frequency)) # sine waves
            # damp = self.damping(o.frequency)
            # e = Gamma(-self.damping(o.frequency)[0], 1.0/max(float(o.frequency)/100.0, 1.0)) # sine waves
            
            if self.rand_phase:
                frames += o[iter] * random_phase() * e[iter]    # Move phases to Osc/Frequency!!!
            else:
                frames += o[iter] * e[iter]

        if self.sustain != None:
            sus_damping = lambda f, a=1.0: -2*np.log2(float(f))/5.0
            #sus_damping = lambda f: -0.5
            self.sustained = self.sustained or Exponential(sus_damping(self.frequency))
            if isinstance(iter, slice):
                frames *= self.sustained[ slice(*list(np.array(iter.indices(iter.stop)) - self.sustain)) ]
            elif isinstance(iter, np.ndarray):
                frames *= self.sustained[ iter - self.sustain ]
            else:
                raise exceptions.NotImplementedError(
                    "Sustain with objects of type %s not implemented yet." % type(iter)
                )
        
        return frames / self.limit # normalize using a single value for whole sound!

    def __repr__(self):
        return "%s(sndobj=%s, n=%s, func=%s, damping=%s, rand_phase=%s>" % \
                (self.__class__.__name__, self.base, self.n, self.func, self.damping, self.rand_phase)

    def __str__(self):
        return "<%s: sndobj=%s, limit=%s, frequency=%s, overtones=%s, func=%s, damping=%s>" % \
                (self.__class__.__name__, self.base, self.limit, self.frequency, self.overtones, self.func, self.damping)


class Multiosc(Overtones):
    # MAKE A MULTIOSC without ENV, iow. sample using overtones and apply_along_axis with sum!!!!
    # ratios = map(lambda r: Fraction.from_float(r).limit_denominator(sampler.rate), h.ratio*h.overtones)
    # samples = map(Frequency.rads, ratios)
    # map(len, samples)
    # Out[58]: [1960, 980, 1960, 490, 393, 980, 280]
    # In [60]: 7*280*393.0 / np.array(map(len, samples))
    # Out[60]: array([  393.,   786.,   393.,  1572.,  1960.,   786.,  2751.])

    # In [11]: np.arange(7) * o.ratio
    # Out[11]: array([0, 11/735, 22/735, 11/245, 44/735, 11/147, 22/245], dtype=object)

    # np.exp(1j * np.atleast_2d(h.overtones).T * np.atleast_2d(2.0 * np.pi * (np.arange(0,44100.0/25) * float(o.ratio))))

    # The problem is that different frequencies have different periods, and getting samples (mod period) would be needed
    # to be implemented at numpy array level...

    @staticmethod
    @memoized
    def angles(ratio, limit):
        if ratio == 0:
            return np.array([0.], dtype=np.float64)
        pi2 = 2 * np.pi
        return pi2 * ratio.numerator * np.arange(0, 1, 1.0/ratio.denominator, dtype=np.float64)

    @staticmethod
    @memoized
    def circle(ratio):
        return np.exp(1j * pi2 * Frequency.angles(ratio))

    def sample(self, iter):
        frames = np.zeros(len(iter), dtype=complex)

        for o in self.oscs:
            if o.frequency == 0: break

            if self.rand_phase:
                frames += o[iter] * random_phase()
            else:
                frames += o[iter]

        return normalize(frames)

