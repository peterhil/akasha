#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Harmonic overtones module
"""

import exceptions
import numpy as np

from akasha.audio.envelope import Exponential
from akasha.audio.frequency import Frequency, FrequencyRatioMixin
from akasha.audio.generators import Generator
from akasha.audio.oscillator import Osc

from akasha.timing import sampler
from akasha.utils.decorators import memoized
from akasha.utils.math import random_phase, map_array, normalize, pi2


class Overtones(FrequencyRatioMixin, Generator):
    """Harmonical overtones for a sound object having a frequency"""

    def __init__(
            self,
            sndobj=Osc(216.0),
            n=8,
            func=lambda x: 1 + x,
            damping=None,
            rand_phase=False):
        super(self.__class__, self).__init__()
        self.base = sndobj
        # TODO Setting ovt.frequency (ovt._hz) leaves ovt.base.frequency (ovt.base._hz)
        # where it was -- is this the desired behaviour?
        self._hz = self.base.frequency
        self.n = n
        self.func = func
        # Sine waves FIXME: separate freq. damping from rate
        # self.damping = damping or (
        #     lambda f, a=1.0: (
        #         -5 * np.log2(float(f)) / (10.0),
        #         a * float(self.frequency)/float(f)
        #         )
        #     )
        #self.damping = damping or (lambda f, a=1.0: -5*np.log2(float(f))/10.0)
        self.damping = damping or (lambda f, a=1.0: -5 * np.log2(float(f)) / 1000.0)
        self.sustain = None
        self.sustained = None
        self.rand_phase = rand_phase

    @property
    def max_overtones(self):
        """Maximum number of overtones to generate for a frequency."""
        low_freq_overtone_limit = 10
        return int(sampler.rate / (2.0 * max(self.frequency, low_freq_overtone_limit)))

    @property
    def limit(self):
        """Get the number of overtones to generate for a frequency."""
        return max(min(self.max_overtones, self.n), 1)

    @property
    def overtones(self):
        """
        Generate overtones using the function given in init.
        The number of overtones is limited by self.limit.
        """
        return np.apply_along_axis(self.func, 0, np.arange(0, self.limit, dtype=np.float64))

    @property
    def oscs(self):
        """Property to get oscillators."""
        # TODO cleanup - make an interface for different Oscs!
        return self.gen_oscs()

    def gen_oscs(self):
        """Generate oscillators based on overtones."""
        base = self.base.__class__
        overtones = np.array(float(self.frequency) * self.overtones, dtype=np.float64)
        if 'Super' == self.base.curve.__class__.__name__:
            oscs = map_array(lambda f: base(f, curve=self.base.curve), overtones, 'vec')
        else:
            oscs = map_array(base, overtones, 'vec')
        return oscs[np.nonzero(oscs)]

    def sample(self, iter):
        """Sample the overtones."""
        if isinstance(iter, int):
            frames = np.array([0j])
        else:
            frames = np.zeros(len(iter), dtype=complex)

        for o in self.oscs:
        # for f in self.overtones:
            # o = deepcopy(self.base)     # Uses deepcopy to preserve superness & other attrs
            # o.frequency = Frequency(self.frequency * f)
            #logger.boring("Overtone frequency: %s" % o.frequency)
            if o.frequency == 0:
                break

            # square waves
            # e = Exponential(0, amp=float(self.frequency / o.frequency * float(self.frequency)))

            # triangle waves
            # e = Exponential(0, amp=float(self.frequency ** 2 / o.frequency ** 2 * float(self.frequency)))

            # sine waves
            # e = Exponential(-o.frequency / 100.0)
            e = Exponential(self.damping(o.frequency))
            # damp = self.damping(o.frequency)
            # e = Gamma(-self.damping(o.frequency)[0], 1.0 / max(float(o.frequency) / 100.0, 1.0))

            if self.rand_phase:
                frames += o[iter] * random_phase() * e[iter]  # Move phases to Osc/Frequency!
            else:
                frames += o[iter] * e[iter]

        if self.sustain is not None:
            sus_damping = lambda f, a = 1.0: -2 * np.log2(float(f)) / 5.0
            #sus_damping = lambda f: -0.5
            self.sustained = self.sustained or Exponential(sus_damping(self.frequency))
            if isinstance(iter, slice):
                frames *= self.sustained[slice(*list(np.array(iter.indices(iter.stop)) - self.sustain))]
            elif isinstance(iter, np.ndarray):
                frames *= self.sustained[iter - self.sustain]
            else:
                raise exceptions.NotImplementedError(
                    "Sustain with objects of type %s not implemented yet." % type(iter)
                )

        return frames / self.limit  # normalize using a single value for whole sound!

    def __repr__(self):
        return "%s(sndobj=%s, n=%s, func=%s, damping=%s, rand_phase=%s>" % \
            (self.__class__.__name__, self.base, self.n, self.func, self.damping, self.rand_phase)

    def __str__(self):
        return "<%s: sndobj=%s, limit=%s, frequency=%s, overtones=%s, func=%s, damping=%s>" % \
            (self.__class__.__name__, self.base, self.limit, self.frequency,
             self.overtones, self.func, self.damping)


class Multiosc(Overtones):
    """Multifrequency oscillator."""
    # MAKE A MULTIOSC without ENV, iow. sample using overtones and apply_along_axis with sum!!!!
    # ratios = map(lambda r: Fraction.from_float(r).limit_denominator(sampler.rate), h.ratio*h.overtones)
    # samples = map(Frequency.rads, ratios)
    # map(len, samples)
    # Out[58]: [1960, 980, 1960, 490, 393, 980, 280]
    # In [60]: 7*280*393.0 / np.array(map(len, samples))
    # Out[60]: array([  393.,   786.,   393.,  1572.,  1960.,   786.,  2751.])

    # In [11]: np.arange(7) * o.ratio
    # Out[11]: array([0, 11/735, 22/735, 11/245, 44/735, 11/147, 22/245], dtype=object)

    # np.exp(1j * np.atleast_2d(h.overtones).T * \
    #     np.atleast_2d(2.0 * np.pi * (np.arange(0,44100.0/25) * float(o.ratio))))

    # The problem is that different frequencies have different periods, and getting
    # samples (mod period) would be needed to be implemented at numpy array level...

    @staticmethod
    @memoized
    def angles(ratio):
        """Frequency angles"""
        if ratio == 0:
            return np.array([0.], dtype=np.float64)
        return pi2 * ratio.numerator * np.arange(0, 1, 1.0 / ratio.denominator, dtype=np.float64)

    @staticmethod
    @memoized
    def circle(ratio):
        """The circle curve for an oscillator at a ratio."""
        return np.exp(1j * pi2 * Frequency.angles(ratio))

    def sample(self, iter):
        """Sample multifrequency oscillator."""
        frames = np.zeros(len(iter), dtype=complex)

        for o in self.oscs:
            if o.frequency == 0:
                break

            if self.rand_phase:
                frames += o[iter] * random_phase()
            else:
                frames += o[iter]

        return normalize(frames)

