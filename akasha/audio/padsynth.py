#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
The public domain PADsynth algorithm from ZynAddSubFx

http://zynaddsubfx.sourceforge.net/doc/PADsynth/PADsynth.htm
http://zynaddsubfx.sourceforge.net/doc_0.html
"""

from __future__ import division

import numpy as np
import scipy as sc

from itertools import izip

from akasha.audio.frequency import Frequency
from akasha.math import normalize, random_phasor


class GaussianCurve(object):

    def __init__(self, frequency, sigma, scale=1.0, base=0.0):
        self.frequency = frequency
        self.sigma = sigma
        self.scale = scale
        self.base = base

    def __call__(self, x):
        return self.scale * np.exp(-((x - self.frequency) ** 2.0) / (2.0 * self.sigma ** 2.0)) + self.base

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(self.__class__.__name__, self.frequency, self.sigma, self.scale, self.base)


class GaussianFrequencyCurve(object):

    def __init__(self, freqs, sigmas):
        self.gaussians = [GaussianCurve(f, s) for f, s in izip(freqs, sigmas)]

    def __call__(self, at):
        return np.apply_along_axis(np.sum, 0, [g(np.asanyarray(at)) for g in self.gaussians])

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.gaussians)


def random_phases(reals):
    return reals * random_phasor(len(reals), 1)


class Padsynth(object):

    def __init__(self, freqs, sigmas):
        self.curve = GaussianFrequencyCurve(freqs, sigmas)

    def __call__(self, at):
        return normalize(sc.ifft(random_phases(self.curve(np.asanyarray(at)))))
