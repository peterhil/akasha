#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mathematical utility functions module.
"""

import cmath
import exceptions
import numpy as np
import sys

if sys.version_info >= (2, 7):
    from collections import OrderedDict
else:
    from ordereddict import OrderedDict
from fractions import Fraction

from akasha.utils.log import logger

from akasha.funct import blockwise
from akasha.timing import sampler


pi2 = np.pi * 2.0


def to_phasor(x):
    """
    Convert complex number to phasor tuple with magnitude and angle (in degrees).
    """
    return np.array([np.abs(x), (np.angle(x) / pi2 * 360)]).T


def nth_root(n):
    """
    Return nth primitive root of unity - a complex number
    located on the 1/nth tau angle on the unit circle.

    http://en.wikipedia.org/wiki/Roots_of_unity
    """
    return np.exp(1j * pi2 * 1.0 / n)


def deg(radians):
    """Degrees to radians conversion."""
    return 180 * (radians / np.pi)


def rad(degrees):
    """Radians to degrees conversion."""
    return np.pi * (degrees / 180.0)


# Utils for frequency ratios etc...

def logn(x, base=np.e):
    """Logarithm of x on some base."""
    return np.log2(x) / np.log2(base)


def roots_periods(base, limit=44100.0):
    """
    Enumerates the roots for base from 0 to limit.

    For example:
    >>> roots_periods(2)
    array([     1.,      2.,      4.,      8.,     16.,     32.,     64.,    128.,
              256.,    512.,   1024.,   2048.,   4096.,   8192.,  16384.,  32768.])

    >>> 44100 / roots_periods(2)
    array([ 44100.             ,  22050.             ,  11025.             ,
             5512.5            ,   2756.25           ,   1378.125          ,
              689.0625         ,    344.53125        ,    172.265625       ,
               86.1328125      ,     43.06640625     ,     21.533203125    ,
               10.7666015625   ,      5.38330078125  ,      2.691650390625 ,
                1.3458251953125])
    """
    # FIXME What this really does?
    # TODO Write a test to ensure it does it right!
    if (base <= 1):
        raise(ValueError("Base can not be less than or equal to one."))
    ex = np.ceil(logn(limit, base))
    return base ** np.arange(ex)


def roots_counts(base, limit=44100.0):
    """
    >>> roots_counts(2)
    array([ 0.000030517578125,  0.00006103515625 ,  0.0001220703125  ,
            0.000244140625   ,  0.00048828125    ,  0.0009765625     ,
            0.001953125      ,  0.00390625       ,  0.0078125        ,
            0.015625         ,  0.03125          ,  0.0625           ,
            0.125            ,  0.25             ,  0.5              ,  1.               ])

    >>> roots_counts(2) * 44100
    array([     1.3458251953125,      2.691650390625 ,      5.38330078125  ,
               10.7666015625   ,     21.533203125    ,     43.06640625     ,
               86.1328125      ,    172.265625       ,    344.53125        ,
              689.0625         ,   1378.125          ,   2756.25           ,
             5512.5            ,  11025.             ,  22050.             ,
            44100.             ])
    """
    # FIXME What this really does? It's not used anywhere anymore.
    # TODO Write a test to ensure it does it right!
    ex = np.floor(logn(limit, base))
    return roots_periods(base, limit) / base ** float(ex)


# Floats

def minfloat(guess=1.0):
    """
    >>> minfloat(1.0)   # minimum positive value of a float
    (5e-324, 1074)

    >>> minfloat(-1.0)  # minimum negative value of a float
    (-5e-324, 1074)

    From: http://seun-python.blogspot.com/2009/06/floating-point-min-max.html
    """
    i = 0
    while(guess * 0.5 != 0):
        guess = guess * 0.5
        i += 1
    return guess, i


def maxfloat(guess=1.0):
    """
    >>> maxfloat(1.0)   # maximum positive value of a float
    (inf, 1024)

    >>> maxfloat(-1.0)  # maximum negative value of a float
    (-inf, 1024)

    From: http://seun-python.blogspot.com/2009/06/floating-point-min-max.html
    """
    guess = float(guess)
    i = 0
    while(guess * 2 != guess):
        guess = guess * 2
        i += 1
    return guess, i


# Arrays

def map_array(func, arr, method='numpy', dtype=object):
    """
    Map over an array pointwise with a function.

    Parameters:
    -----------
    func : function
        Function of one argument
    arr : ndarray
        Input array
    method : str
        One of [ 'numpy' | 'vec' | 'map' ]
    dtype : Numpy dtype or str
        Datatype
    """
    shape = arr.shape
    if method == 'numpy':
        res = np.apply_along_axis(func, 0, arr.flat)
    elif method == 'vec':
        vf = np.vectorize(func)
        res = vf(arr.flat)
    elif method == 'map':
        res = np.array(map(func, arr.flat), dtype=dtype)  # pylint: disable=W0141
    else:
        raise exceptions.NotImplementedError("map_array(): method '{0}' missing.".format(method))
    return res.reshape(shape)


def as_complex(a):
    """
    Convert real number coordinate points to complex samples.
    """
    return a.transpose().flatten().view(np.complex128)


def complex_as_reals(samples):
    """
    Convert complex samples to real number coordinate points.
    """
    return samples.view(np.float64).reshape(len(samples), 2).transpose()    # 0.5 to 599.5


def find_closest_index(arr, target):
    """
    Finds the index of the first item in array 'arr', which
    has the absolute value closest to 'target'.
    """
    return np.argmin(np.abs(np.atleast_1d(arr) - target))


def blockiter(snd):
    """
    Iterate sound using blocks of size sampler.blocksize().
    """
    return blockwise(snd, sampler.blocksize())


# Random

def rand_between(inf, sup, n=1, random=np.random.random):
    """
    Generate n random numbers using given function (defaults to np.random.random())
    on the half open interval [inf, sup).
    """
    return np.atleast_1d((sup - inf) * random(n) + inf)


def random_phase(n=1, random=np.random.random):
    """
    Generate n complex numbers on the unit circle with random phase (angle)
    using the function given (defaults to np.random.random()).
    """
    return np.atleast_1d(cmath.rect(1.0, pi2 * random(n) - np.pi))


# Primes

def primes(min, max):
    """
    Primes in the range min, max. Returns a numpy array.
    """
    primes = [2]
    i = 3
    while i <= max:
        for p in primes:
            if i % p == 0 or p * p > i:
                break
        if i % p != 0:
            primes.append(i)
        i = i + 2
    return np.array(primes)



def gcd(a, b):
    """
    Greatest common divisor of a and b.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Least common multiple of a and b.
    """
    return a * b / gcd(a, b)


# Vectorized functions

np.gcd = lambda a, axis = None: reduce(gcd, a)
np.lcm = lambda a, axis = None: reduce(lcm, a)

np.getattr = np.vectorize(lambda x, attr: getattr(x, attr), otypes=['object'])


def as_fractions(a, limit=1000000):
    from_float = np.vectorize(lambda y: Fraction.from_float(y).limit_denominator(limit))
    return from_float(a)


# Factors

def sq_factors(n):
    """
    Find factors of n, by trying to divide with integers between 1 and sqrt(n).
    If the modulus is zero, it's a factor.
    Note! This only returns factors less or equal to sqrt(n)!
    """
    try_divisors = np.arange(1, np.sqrt(np.abs(n) + 1), dtype=np.int)
    # TODO: find complex zeros with conjugates!!!
    z = np.ma.masked_not_equal(float(n) / try_divisors % 1, 0)
    return np.ma.masked_array(try_divisors, z.mask).compressed().astype(np.int)


def factors(n):
    assert np.all(np.equal(np.mod(n, 1), 0)), "Value %s is not an integer!" % n
    f = sq_factors(n)
    return np.unique(np.append(f, np.array(n, dtype=np.int) / f[::-1]))


def arr_factors(arr, method='map'):
    return map_array(factors, arr, method=method, dtype=object)


def get_factorsets(n):
    return dict(enumerate(map(lambda x: set(factors(x)), np.arange(n + 1))))


def factor_supersets(factors_in, redundant=None, limit=None):
    lim = limit if limit else len(factors_in) - 1  # Change if factors_in doesn't include n + 1!
    length_of_value = lambda x: len(x[1])

    ess = OrderedDict(sorted(factors_in.items(), key=length_of_value, reverse=True))
    red = redundant if redundant else OrderedDict()
    #logger.info("STARTING\tEssential keys: %s redundant keys: %s" % (ess.keys(), red.keys()))

    for (j, fset) in filter(lambda y: 0 <= y[0] < lim, ess.items()):
        ind = (lim, j)
        logger.debug("\t#%s" % (ind, ))
        if fset.issubset(factors_in[lim]):
            msg = "\t#%s:\tSet %s is subset of %s, will add missing factors of %s to redundant"
            logger.info(msg % (ind, fset, factors_in[lim], j))
            msg = "#%s:\tMoving %s from essential to redundant. (factors in %s)"
            logger.warn(msg % (ind, j, factors_in[j]))
            if j in ess:
                red[j] = ess.pop(j)
            for f in fset:
                if f in ess:
                    red[f] = ess.pop(f)
                # if not f in red:
                #     (ess, re2) = factor_supersets(ess, red, limit=lim-1)
                #     for k in re2.keys():
                #         if (not k in red) and k in ess:
                #             msg = "\t#%s:\t\tMoving %s to redundant. (factors in %s)"
                #             logger.warn(msg % (ind, k, factors_in[k]))
                #             if k in ess:
                #                 red[k] = ess.pop(k)  #es2[k]
    #logger.debug("\t#%s:\tEssential keys: %s redundant keys: %s" % (ind, ess.keys(), red.keys()))

    assert set(ess).isdisjoint(red), "Redundant values {} contained in essential".format(
        set(ess).intersection(red))
    (ess, red) = factor_supersets(ess, red, limit=lim - 1)
    assert set(ess).isdisjoint(red), "Redundant values {} contained in essential".format(
        set(ess).intersection(red))
    logger.debug("This should be empty set: %s" % set(ess).intersection(red))
    return (ess, red)


# Signal processing utils

def identity(x):
    return x


def pcm(snd, bits=16, axis='imag'):
    #if isinstance(snd[0], np.floating): axis = 'real'
    return np.cast['int' + str(bits)](getattr(snd, axis) * (2 ** bits / 2.0 - 1))


def normalize(signal):
    max = np.max(np.abs(signal))
    if not np.all(np.isfinite(signal)):
        logger.debug("Normalize() substituting non-numeric max: %s" % max)
        signal = np.ma.masked_invalid(signal).filled(0)
        max = np.max(np.abs(signal))
        logger.debug("Normalize() substituted max: %s" % max)
    if max == 0:
        logger.debug("Normalize() got silence!" % max)
        return signal
    #logger.debug("Normalize() by a factor: %s" % max)
    return signal / max


def clip(signal, inplace=False):
    """
    Clips complex samples to unit area (-1-1j, +1+1j).
    """
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal)
    if not inplace:
        signal = signal.copy()
    reals = signal.view(np.float)
    np.clip(reals, a_min=-1, a_max=1, out=reals)    # Uses out=reals to transform in-place!
    return signal


def pad(signal, index=-1, count=1, value=None):
    """
    Inserts a padding value at index repeated count number of times.
    If value is None, uses an index from signal.
    """
    length = len(signal)
    space = index % (length + 1)
    value = (value if value is not None else signal[index])
    return np.concatenate((signal[0:space], np.repeat(value, count), signal[space:length]))


def distances(signal):
    # See also np.diff!
    if hasattr(signal, '__len__') and (len(signal) >= 2):
        return np.abs(signal[1:] - signal[:-1])
    else:
        raise ValueError("Expected signal to have at least two samples. Signal was: %s" % signal)


def diffs(signal, start=0, end=0):
    # Could use np.apply_over_axes - profile with time?
    # There is also np.ediff1d!!!
    return np.append(start, signal[1:]) - np.append(signal[:-1], end)


def get_zerocrossings(signal):
    peaks = pad(distances(np.angle(signal) / np.pi), 0)
    res = np.round((peaks - peaks[1]) * 1.1) * np.sign(np.angle(signal))
    return res


def get_impulses(signal, tau=False):
    if tau:
        peaks = pad(distances(np.angle(signal) / np.pi % 2), 0)
        res = np.fmax(np.sign((peaks - 1)) * 2, 0)
    else:
        peaks = pad(distances(np.angle(signal) % pi2), 0)
        res = np.fmax(np.sign((peaks - np.pi)) * pi2, 0)
    return res
