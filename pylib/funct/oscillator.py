#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np
import operator
from fractions import Fraction

# My modules
from timing import Sampler, samples, times

# Utils
# from utils.math import *
# from utils.graphing import *
from utils.audio import play, write

# Functional
import itertools as itr
import functools as fun
from xoltar import lazy
from xoltar.functional import *

# Settings
np.set_printoptions(precision=16, suppress=True)


### Limiter functions

def step_function(op=operator.le, limit=0, default=0):
    def fn(value):
        other = limit() if callable(limit) else limit
        return value if op(value, other) else default
    return fn

def limit_below(limit, default=0):
    """Make a brick wall low pass filter.

    Example:
    =======
    >>> lp = limit_below(7)
    >>> lp(7.1)
    0
    >>> lp(6.5)
    6.5
    """
    return step_function(operator.le, limit, default=0)

def limit_negative(f):
    fn = step_function(operator.ge, 0, 0)
    return fn(f)

def nyquist(ratio):
    """Limit ratio on the normalized Nyquist Frequency band: Fraction(-1, 2)..Fraction(1, 2)

    >>> nyquist(Fraction(22050, 44100))
    Fraction(1, 2)

    >>> nyquist(Fraction(22051, 44100))
    Fraction(0, 1)

    >>> nyquist(Fraction(-5, 8))
    Fraction(0, 1)
    """
    assert isinstance(ratio, Fraction), "Ratio should be a Fraction, got %s" % type(ratio).__name__
    fn = step_function(operator.le, limit=Fraction(1, 2), default=Fraction(0, 1))
    return fn(np.abs(ratio))

def wrap(f, modulo=1):
    """Wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.

    Examples:
    =========

    >>> wrap(Fraction(9, 8))
    Fraction(1, 8)

    >>> wrap(Fraction(-1, 8))
    Fraction(7, 8)
    """
    return f % modulo


### Frequencies

def limit_resolution(f, max=Sampler.rate):
    return Fraction(int(round(f * max)), max)

def hz(f, fs=Sampler.rate, rounding='native'):
    """Return normalized frequency (as a Fraction) from physical frequency.

    Examples:
    =========

    >>> Sampler.rate = 44100

    >>> hz(5512.5)
    Fraction(1, 8)

    >>> hz(20.0)
    Fraction(1, 2205)

    >>> float(hz(20.0) * Sampler.rate)
    20.0

    # A different sampling rate can be given with the fs option:
    >>> hz(88.0, 48000)
    Fraction(11, 6000)
    """
    ratio = Fraction.from_float(float(f)/fs)
    if rounding == 'native':
        ratio = ratio.limit_denominator(fs)
    else:
        ratio = limit_resolution(ratio, fs).limit_denominator(fs)
    return np.array(ratio)


### Generators

def accumulator(n):
    """Function object using closure, see:
    http://en.wikipedia.org/wiki/Function_object#In_Python"""
    def inc(x):
        inc.n += x
        return inc.n
    inc.n = n
    return inc

def osc(freq):
    def osc(times):
        osc.gen = 1j * 2 * np.pi
        return np.exp( osc.gen * osc.freq * (times % (1.0 / osc.freq)) )
    osc.freq = freq
    def ratio():
        return osc.freq / float(Sampler.rate)
    osc.ratio = ratio
    return osc

def exp(rate):
    def exp(times):
        return np.exp( exp.rate * times )
    exp.rate = rate
    return exp

if __name__ == "__main__":
    import doctest
    doctest.testmod()
