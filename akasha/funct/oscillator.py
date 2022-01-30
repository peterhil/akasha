#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Functional oscillator module using functor objects.
"""

from __future__ import division

import numpy as np
import operator

from fractions import Fraction

from akasha.timing import sampler
from akasha.math import pi2


# Limiter functions


def step_function(op=operator.le, limit=0, default=0):
    """
    Limit the range of the function with an operator function and
    a limit (number or callable). Return the default value outside the range.
    """

    def fn(value):
        """
        Calculate the result from step_function().
        """
        other = limit() if callable(limit) else limit
        return value if op(value, other) else default

    return fn


def limit_below(limit, default=0):
    """
    Make a brick wall low pass filter.

    Example:
    =======
    >>> lp = limit_below(7)
    >>> lp(7.1)
    0
    >>> lp(6.5)
    6.5
    """
    return step_function(operator.le, limit, default)


def limit_negative(f):
    """
    Limit negative values to zero.
    """
    fn = step_function(operator.ge, 0, 0)
    return fn(f)


def nyquist(ratio):
    """
    Limit ratio on the normalized Nyquist Frequency band:
    Fraction(-1, 2)..Fraction(1, 2)

    >>> nyquist(Fraction(22050, 44100))
    Fraction(1, 2)

    >>> nyquist(Fraction(22051, 44100))
    Fraction(0, 1)

    >>> nyquist(Fraction(-5, 8))
    Fraction(0, 1)
    """
    assert isinstance(
        ratio, Fraction
    ), f'Ratio should be a Fraction, got {type(ratio).__name__}'
    fn = step_function(
        operator.le, limit=Fraction(1, 2), default=Fraction(0, 1)
    )
    return fn(np.abs(ratio))


def wrap(f, modulo=1):
    """
    Wrap roots: 9/8 == 1/8 in Osc! This also helps with numeric accuracy.

    Examples:
    =========

    >>> wrap(Fraction(9, 8))
    Fraction(1, 8)

    >>> wrap(Fraction(-1, 8))
    Fraction(7, 8)
    """
    return f % modulo


# Frequencies


def limit_resolution(freq, limit=sampler.rate):
    """
    Limit frequency resolution.
    """
    return Fraction(int(round(freq * limit)), limit)


def hz(f, fs=sampler.rate, rounding='native'):
    """
    Return normalized frequency (as a Fraction) from physical frequency.

    Examples:
    =========

    >>> sampler.rate = 44100

    >>> hz(5512.5)
    Fraction(1, 8)

    >>> hz(20.0)
    Fraction(1, 2205)

    >>> float(hz(20.0) * sampler.rate)
    20.0

    # A different sampling rate can be given with the fs option:
    >>> hz(88.0, 48000)
    Fraction(11, 6000)
    """
    ratio = Fraction.from_float(float(f) / fs)
    if rounding == 'native':
        ratio = ratio.limit_denominator(fs)
    else:
        ratio = limit_resolution(ratio, fs).limit_denominator(fs)
    return np.array(ratio)


# Generators


def accumulator(n):
    """
    Function object using closure, see:
    http://en.wikipedia.org/wiki/Function_object#In_Python
    """

    def inc(x):
        """
        Increment accumulator.
        """
        inc.n += x
        return inc.n

    inc.n = n
    return inc


def osc(freq):
    """
    Oscillator functor.
    """

    def at(times):
        """
        Sample oscillator at times.
        """
        return np.exp(1j * pi2 * freq * (times % 1.0))

    at.freq = freq

    def ratio():
        """
        The frequency ratio of the oscillator.
        """
        return freq / float(sampler.rate)

    at.ratio = ratio
    return at


def exp(rate):
    """
    Exponential envelope for a rate.
    """

    def at(times):
        """
        Sample exponential at times.
        """
        return np.exp(rate * times)

    at.rate = rate
    return at


if __name__ == '__main__':
    import doctest

    doctest.testmod()
