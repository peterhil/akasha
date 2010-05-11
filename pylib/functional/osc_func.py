#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import types
from cmath import rect, polar, phase, pi, exp
from fractions import Fraction
# My modules
# from generators import PeriodicGenerator
from timing import Sampler

import itertools as itr
import functools as fun
from xoltar import lazy
from xoltar.functional import *

np.set_printoptions(8,1000,10,suppress=True)

# General flow:
# 1. Get time slice
# 2. Convert to frames
# 3. Apply frames to generating function
# 4. ?

### Time related functions ###
from datetime import timedelta

td = np.timedelta64

us = lambda t: td(t)            # µs, microseconds
ms = lambda t: us(t * 10**3)    # milliseconds
sec = lambda t: timedelta(seconds=t)
minutes = lambda t: td(timedelta(minutes=t))

Hz = lambda x: Fraction(1, x)

def timeslice(iterable, unit=sec):
    """Convert frame numbers to time.

    In [10]: timeslice([0, 8125, 44100, 44100*500.12])
    Out[10]: array([0:00:00, 0:00:00.184240, 0:00:01, 0:08:20.120000], dtype=timedelta64[us])
    """
    result = np.divide(np.array(iterable), float(Sampler.rate))
    return np.fromiter(imap(unit, result), dtype=td)

def frames(iterable):
    """Convert time deltas to frame numbers (ie. 1.0 => 44100)"""
    # iterable = (imap(sec, iterable))
    result = np.multiply(np.array(iterable), Sampler.rate)
    result = np.fromiter(imap(sec, result), dtype=np.float64)
    return np.cast['uint64'](np.round(result))


### Sampling ###

def sample(iterable, *times):
    return np.fromiter(islice(iterable, *(frames(times))), dtype=np.complex64)

### Generating functions ###

# In [69]: v_root = np.frompyfunc(nth_root, 1,1)
# 
# In [70]: l = nth_root(np.arange(1,20))
# 
# In [71]: l
# Out[71]: 
# array([  1.00000000e+00 -2.44929360e-16j,
#         -1.00000000e+00 +1.22464680e-16j,
#         -5.00000000e-01 +8.66025404e-01j,
#          6.12323400e-17 +1.00000000e+00j,
#          3.09016994e-01 +9.51056516e-01j,
#          5.00000000e-01 +8.66025404e-01j,
#          6.23489802e-01 +7.81831482e-01j,
#          7.07106781e-01 +7.07106781e-01j,
#          7.66044443e-01 +6.42787610e-01j,
#          8.09016994e-01 +5.87785252e-01j,
#          8.41253533e-01 +5.40640817e-01j,
#          8.66025404e-01 +5.00000000e-01j,
#          8.85456026e-01 +4.64723172e-01j,
#          9.00968868e-01 +4.33883739e-01j,
#          9.13545458e-01 +4.06736643e-01j,
#          9.23879533e-01 +3.82683432e-01j,
#          9.32472229e-01 +3.61241666e-01j,
#          9.39692621e-01 +3.42020143e-01j,   9.45817242e-01 +3.24699469e-01j])
# 
# In [72]: %timeit l = np.fromiter(imap(v_root, np.arange(1,1000)), dtype=np.complex)
# 10 loops, best of 3: 52.8 ms per loop

def angfreq(f):
    return 2 * pi * f

class oscf(Functor):
    
    def __call__(self, *args):
        return np.exp( 1j*2*pi * (args % 1.0))

def oscillate(phases):
    """
    In [18]: oscillate(np.arange(0,8) * 1.0/44100 * 5512.5)
    Out[18]: 
    array([ 1.00000000+0.j        ,  0.70710678+0.70710678j,
            0.00000000+1.j        , -0.70710678+0.70710678j,
           -1.00000000+0.j        , -0.70710678-0.70710678j,
           -0.00000000-1.j        ,  0.70710678-0.70710678j])
    
    In [20]: 5512.5/44100.0
    Out[20]: 0.125
    
    In [21]: 5512.5 * 1.0/44100.0
    Out[21]: 0.125
    
    In [35]: 5512.5/44100.0
    Out[35]: 0.125
    
    
    np.exp( 1j*2*pi * 5512.5 * Fraction(1,44100) * np.arange(0,8))
    np.exp( 1j*2*pi * (np.linspace(0,1,8,endpoint=False) % 1.0))
    
    """
    return np.exp( 1j*2*pi * (phases % 1.0))

def osc_freq(iterable, frq):
    return oscillate(iterable * frq * 1.0/Sampler.rate)

def osc(n):
    return itr.cycle(roots(n))

def freq(fl):
    """Calculate ratio for a frequency."""
    ratio = Fraction.from_float(float(fl)/Sampler.rate).limit_denominator(Sampler.rate)
    return ratio

def freq2(fl):
    ratio = Fraction(*(fl/Sampler.rate).as_integer_ratio()).limit_denominator(Sampler.rate)
    return ratio

def roots(period):
    """Calculate n principal roots of unity."""
    return np.exp(np.linspace(0, 2 * pi, period, endpoint=False) * 1j)

def nth_root(n):
    """Calculate principal nth root of unity."""
    return np.exp(1.0/n * np.pi * 2 * 1j)

def to_phasor(x):
    return (abs(x), (phase(x) / (2 * pi) * 360))

def reorder(arr, n):
    return np.arange(self.period) * self.order % self.period
