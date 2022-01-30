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

from builtins import zip

from akasha.audio.frequency import Frequency
from akasha.math import normalize, random_phasor
from akasha.utils.python import class_name


class GaussianCurve:
    def __init__(self, frequency, sigma, scale=1.0, base=0.0):
        self.frequency = frequency
        self.sigma = sigma
        self.scale = scale
        self.base = base

    def __call__(self, x):
        gaussian = -((x - self.frequency) ** 2.0) / (2.0 * self.sigma ** 2.0)
        return self.scale * np.exp(gaussian) + self.base

    def __repr__(self):
        return (
            f'{class_name(self)}({self.frequency!r}, {self.sigma!r}, '
            + f'{self.scale!r}, {self.base!r})'
        )


class GaussianFrequencyCurve:
    def __init__(self, freqs, sigmas):
        self.gaussians = [GaussianCurve(f, s) for f, s in zip(freqs, sigmas)]

    def __call__(self, at):
        partials = [g(np.asanyarray(at)) for g in self.gaussians]
        return np.apply_along_axis(np.sum, 0, partials)

    def __repr__(self):
        return f'{class_name(self)}({self.gaussians!r})'


def random_phases(reals):
    return reals * random_phasor(len(reals), 1)


class Padsynth:
    def __init__(self, freqs, sigmas):
        self.curve = GaussianFrequencyCurve(freqs, sigmas)

    def __call__(self, at):
        curve = self.curve(np.asanyarray(at))
        return normalize(sc.ifft(random_phases(curve)))
