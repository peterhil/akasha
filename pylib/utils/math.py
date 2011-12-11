#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cmath
from fractions import Fraction

from timing import Sampler

sample_rate = Sampler.rate
euler = np.e
debug = False

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

def deg(radians):
    return 180 * (radians / np.pi)

def rad(degrees):
    return np.pi * (degrees / 180.0)

# Utils for frequency ratios etc...

def logn(x, base=euler):
    return np.log2(x) / np.log2(base)

def roots_periods(base, limit=44100.0):
    """Enumerates the roots for base form 0 to limit.
    For example:
    >>> roots_count(2)
    array([ 0.00003052,  0.00006104,  0.00012207,  0.00024414,  0.00048828,
            0.00097656,  0.00195312,  0.00390625,  0.0078125 ,  0.015625  ,
            0.03125   ,  0.0625    ,  0.125     ,  0.25      ,  0.5       ])
    """
    if (base <= 1):
        raise(ValueError("Base can not be less than or equal to one."))
    ex = np.floor(logn(limit,base))
    return base**np.arange(ex)

def roots_counts(base, limit=44100.0):
    ex = np.floor(logn(limit,base))
    return roots_periods(base, limit)/base**float(ex)


# Floats

# Following two methods are modified from:
# http://seun-python.blogspot.com/2009/06/floating-point-min-max.html

def minfloat(guess):
    i = 0
    while(guess * 0.5 != 0):
        guess = guess * 0.5
        i += 1
    return guess, i

def maxfloat(guess = 1.0):
    guess = float(guess)
    i = 0
    while(guess * 2 != guess):
        guess = guess * 2
        i += 1
    return guess, i


# Random

def rand_between(min, max, size=1, random=np.random.random):
    return np.atleast_1d( (max - min) * random(size) + min )

def random_phase(random=np.random.random):
    return np.atleast_1d( cmath.rect(1.0, 2.0 * np.pi * random() - np.pi) )


# Primes

def primes(min, max):
    """Primes in the range min, max. Returns a numpy array."""
    if 2 >= min: print 2
    primes = [2]
    i = 3
    while i <= max:
        for p in primes:
            if i % p == 0 or p * p > i: break
        if i % p != 0:
            primes.append(i)
        i = i+2
    return np.array(primes)


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

def pad(d, index=-1, count=1, value=None):
    """Inserts a padding value at index repeated count number of times. If value is None, uses an index from d."""
    length = len(d)
    space = index % (length + 1)
    value = (value if value != None else d[index])
    return np.concatenate((d[0:space], np.repeat(value, count), d[space:length]))

def distances(signal):
    if hasattr(signal, '__len__') and (len(signal) >= 2):
        return np.abs(signal[1:] - signal[:-1])
    else:
        raise ValueError("Expected signal to have at least two samples to calculate distances. Signal was: %s" % signal)

def diffs(signal, start=0, end=0):
    # Could use np.apply_over_axes - profile with time?
    return np.append(start, signal[1:]) - np.append(signal[:-1], end)
