#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cmath
from fractions import Fraction

# Viewing complex samples as pairs of reals:
# 
# o = Osc(1,8)
# a = o.samples.copy()
# a
# b = a.view(np.float64).reshape(8,2)
# b *= np.array([2,0.5])
# b
# c = b.view(np.complex128).reshape(8,)
# c
# b.transpose()
# b.transpose()[0]

def to_phasor(x):
    """Convert complex number to phasor tuple with magnitude and angle (in degrees)."""
    return (abs(x), (cmath.phase(x) / (2 * cmath.pi) * 360))

def to_phasors(samples):
    return np.array(map(to_phasor, samples))

def nth_root(n):
    return np.exp(1j * 2 * np.pi * 1.0/n)


# Signal processing utils

def normalize(signal):
    max = np.max(np.abs(signal))
    if (max != 0):
        return signal / max
    else:
        return signal # ZeroDivision if max=0!

def clip(signal, inplace=False):
    """Clips complex samples to unit area (-1-1j, +1+1j)."""
    if not inplace:
        signal = signal.copy()
    reals = signal.view(np.float)
    np.clip(reals, a_min=-1, a_max=1, out=reals)    # Uses out=reals to transform in-place!
    return signal