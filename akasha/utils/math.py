#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mathematical utility functions module.
"""

from __future__ import division

import cmath
import exceptions
import numpy as np
import scipy as sc
import sys

if sys.version_info >= (2, 7):
    from collections import OrderedDict
else:
    from ordereddict import OrderedDict
from cmath import rect
from fractions import Fraction

from akasha.funct import blockwise
from akasha.timing import sampler
from akasha.utils.log import logger


pi2 = np.pi * 2.0
cartesian = np.vectorize(rect, doc="""Return cartesian points from polar coordinate radiuses and angles.""")


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


def rad_to_deg(angles):
    """
    Radians to degrees conversion.
    """
    return 180 * (angles / np.pi)


def deg_to_rad(angles):
    """
    Degrees to radians conversion.
    """
    return np.pi * (angles / 180.0)


def tau_to_rad(angles):
    """
    Tau angles to radians conversion.
    """
    return pi2 * angles


def rad_to_tau(angles):
    """
    Radians to tau angles conversion.
    """
    return angles / pi2


# Utils for frequency ratios etc...


def cents(*args):
    """
    Calculate cents from interval or frequency ratio(s).
    When using frequencies, give greater frequencies first.

    >>> cents(float(Fraction(5,4)))
    array([ 386.31371386])

    >>> cents(440/27.5)
    array([ 4800.])     # equals four octaves

    >>> cents(np.arange(8)/16.0+1)
    array([[   0.        ,  104.9554095 ,  203.91000173,  297.51301613,
             386.31371386,  470.78090733,  551.31794236,  628.27434727]])
    """
    return 1200.0 * np.log2(args)


def cents_diff(a, b):
    """
    Calculate the difference of two frequencies in cents."
    """
    return np.abs(cents(float(a)) - cents(float(b)))


def interval(*cnt):
    """
    Calculate interval ratio from cents.

    >>> interval(100)
    array([ 1.05946309])    # one equal temperament semitone

    >>> interval(386.31371386)
    array([ 1.25])          # 5:4, perfect fifth

    >> frac = lambda f: map(Fraction.limit_denominator, map(Fraction.from_float, f))
    >> [frac for i in interval(np.arange(5) * 386.31371386)]
    [[Fraction(1, 1),
      Fraction(5, 4),
      Fraction(25, 16),
      Fraction(125, 64),
      Fraction(625, 256)]]
    """
    return np.power(2, np.asanyarray(cnt) / 1200.0)


def freq_plus_cents(f, cnt):
    """
    Calculate what frequency is given cents apart from a frequency f.
    """
    return f * interval(cnt)


def logn(x, base=np.e):
    """
    Logarithm of x on some base.
    """
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

def map_array(func, arr, method='vec', dtype=None):
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
    # pylint: disable=E1103

    arr = np.atleast_1d(arr)
    shape = arr.shape

    if method == 'numpy':
        res = np.apply_along_axis(np.vectorize(func), 0, arr.flat)
    elif method == 'vec':
        res = np.vectorize(func)(arr.flat)
    elif method == 'map':
        res = np.array(map(func, arr.flat))  # pylint: disable=W0141
    else:
        raise exceptions.NotImplementedError("map_array(): method '{0}' missing.".format(method))

    if dtype is not None:
        res = res.astype(dtype)

    return res.reshape(shape)


def as_complex(a):
    """
    Convert real number coordinate points to complex samples.
    """
    return a.transpose().flatten().view(np.complex128)


def complex_as_reals(signal, dtype=np.float64):
    """
    Convert complex signal to real number coordinate points.
    """
    signal = np.asanyarray(signal, dtype=np.complex128)
    if signal.ndim < 1:
        signal = np.atleast_1d(signal)
    return signal.view(dtype).reshape(len(signal), 2).transpose()


def as_polar(signal, dtype=np.complex128):
    """
    Return a complex signal in polar coordinates.
    """
    return np.array([np.abs(signal), np.angle(signal)], dtype=dtype).T


def as_rect(polar, dtype=np.complex128):
    """
    Return a complex cartesian coordinate signal from polar coordinates.
    """
    magnitude, angle = polar.T
    return np.array(magnitude * np.exp(1j * angle), dtype=dtype)


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


def random_phasor(n=1, amp=1.0, random=np.random.random):
    """
    Generate n complex numbers on the unit circle with random phase (angle)
    using the function given (defaults to np.random.random()).

    Arguments
    =========

    n:
        The number of items to generate
    amp, one of:
        - scalar real value
        - sequence of length n
        - callable of one argument (same as n)
    random:
        - callable of one argument (same as n)

    Example
    =======

    from funckit import xoltar as fx

    rf = fx.curry(np.random.poisson, 100)  # Try different values and random functions
    snd = normalize(random_phase(44100 * 5, amp=rf))
    graph(snd)  # or anim(Pcm(snd))
    """
    if np.isscalar(amp):
        return [cmath.rect(a, b) for a, b in izip(np.repeat(amp, n), pi2 * random(n) - np.pi)]
    elif callable(amp):
        return np.array(map(cmath.rect, *np.array([amp(n), pi2 * random(n) - np.pi])))
    else:
        assert n == len(amp), "Arguments (n, amp) should have the same length, "
        "got: ({0}, {1})".format(n, len(amp))
        return np.array(map(cmath.rect, *np.array([amp, pi2 * random(n) - np.pi])))


# Primes

def primes(inf, sup, dtype=np.uint64):
    """
    Primes in the closed interval [inf, sup].
    Returns a numpy array.
    """
    primes = [2]  # pylint: disable=W0621
    i = 3

    while i <= sup:
        for p in primes:
            if i % p == 0 or p ** 2 > i:
                break
        if i % p != 0:  # pylint: disable=W0631
            primes.append(i)
        i = i + 2

    primes = np.array(primes, dtype=dtype)
    return primes[primes >= inf]


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

np.getattr = np.vectorize(
    lambda x, attr: getattr(x, attr),  # pylint: disable=W0108
    otypes=['object']
)


def as_fractions(arr, limit=1000000):
    """
    Return array as Fractions with denominators up to the limit given.
    """
    from_float = np.vectorize(lambda y: Fraction.from_float(y).limit_denominator(limit))
    return from_float(arr)


# Factors

def sq_factors(n):
    """
    Find factors of integer n, by trying to divide with integers between 1 and sqrt(n).
    If the modulus is zero, it's a factor.
    Note! This only returns factors less or equal to sqrt(n)!
    """
    try_divisors = np.arange(1, np.sqrt(np.abs(n) + 1), dtype=np.int)
    # TODO: find complex zeros with conjugates!!!
    z = np.ma.masked_not_equal(float(n) / try_divisors % 1, 0)
    return np.ma.masked_array(try_divisors, z.mask).compressed().astype(np.int)


def factors(n):
    """
    Return factors of integer n.
    """
    assert np.all(np.equal(np.mod(n, 1), 0)), "Value %s is not an integer!" % n
    f = sq_factors(n)
    return np.unique(np.append(f, np.array(n, dtype=np.int) / f[::-1]))


def arr_factors(arr, method='map'):
    """
    Maps factors() to an array.
    """
    return map_array(factors, arr, method=method, dtype=object)


def get_factorsets(n):
    """
    Get a dict of factors from 0 to integer n as sets.
    """
    # pylint: disable=W0141
    return dict(enumerate(map(lambda x: set(factors(x)), np.arange(n + 1))))


def factor_supersets(factors_in, redundant=None, limit=None):
    """
    Try to find smallest set of integers with most unique factors
    that are supremum to the limit (usually sampling rate).

    Arguments
    =========
    factors_in:
        Should be an output from get_factorsets(),
        that is, a dictionary with integers as keys and their factors in sets as values.

    redundant:
        Can be the redundant factors from a previous run.

    limit:
        Upper limit is either the number given (usually sampling rate),
        or the length of the factors_in.

    Returns the 'essential' set and the redundant factor sets as a tuple.
    """
    # TODO: Refactor factor_supersets!
    lim = limit if limit else len(factors_in) - 1  # Change if factors_in doesn't include n + 1!
    length_of_value = lambda x: len(x[1])

    ess = OrderedDict(sorted(factors_in.items(), key=length_of_value, reverse=True))
    red = redundant if redundant else OrderedDict()
    #logger.info("STARTING\tEssential keys: %s redundant keys: %s" % (ess.keys(), red.keys()))

    for (j, fset) in [item for item in ess.items() if 0 <= item[0] < lim]:
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
    """
    Identity function -- return the argument unchagned. Useful in functional programming.
    """
    return x


def numberof(items):
    """
    Get the number of items.
    If items is scalar, interpret items as a number, otherwise the length of items.
    """
    return items if np.isscalar(items) else len(items)

def pcm(signal, bits=16, axis='imag'):
    """
    Get a pcm sound with integer samples from the complex signal,
    that is playable and usable with most audio libraries.
    """
    #if isinstance(signal[0], np.floating): axis = 'real'
    return np.cast['int' + str(bits)](getattr(signal, axis) * (2 ** bits / 2.0 - 1))


def normalize(signal):
    """
    Normalises signal into interval [-1, +1] and replaces ±NaN and ±Inf values with zeros.
    """
    sup = np.max(np.abs(signal))

    if not np.all(np.isfinite(signal)):
        logger.debug("Normalize() substituting non-numeric max: %s" % sup)
        signal = np.ma.masked_invalid(signal).filled(0)
        sup = np.max(np.abs(signal))
        logger.debug("Normalize() substituted max: %s" % sup)

    if sup == 0:
        logger.debug("Normalize() got silence!" % sup)
        return signal

    return signal / sup


def inside(arr, low, high):
    """
    Return values of array inside interval [low, high]
    """
    return np.ma.masked_outside(arr, low, high).compressed()


def clip(signal, limit=1.0, inplace=False):
    """
    Clips complex signal to unit rectangle area (-1-1j, +1+1j).
    """
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal)

    if not inplace:
        signal = signal.copy()

    reals = signal.view(np.float)
    np.clip(reals, a_min=-limit, a_max=limit, out=reals)  # Do clipping in-place!

    return signal


def repeat(signal, times):
    """
    Repeat the signal the desired number of times.
    """
    signal = np.asarray([signal]).flatten()
    if times < 1:
        return signal[0:0]
    if times == 1:
        return signal
    return np.repeat([signal], times, axis=0).flatten()


def pad(signal, index=-1, count=1, value=None):
    """
    Inserts a padding value at index repeated count number of times.
    If value is None, uses an index from signal.
    """
    signal = np.asarray([signal]).flatten()
    length = len(signal)
    space = index % (length + 1)
    value = (value if value is not None else signal[index])
    return np.concatenate((signal[0:space], repeat(value, count), signal[space:length]))


def pad_minlength(signal, padding, minlength, index=-1, fromleft=False):
    """
    Pad a signal with padding value(s) on the left to be at least the given length.
    """
    padding = np.asanyarray([padding]).flatten()
    if len(padding) == 0:
        padding = [0]
    missing = max(0, minlength - len(signal))
    if not missing:
        return signal
    times = int(np.ceil(missing / len(padding)))  # Will be at least 1

    if not fromleft:  # Take missing items from the head of the repeated padding
        padding = repeat(padding, times)[:missing]
    else:  # Take missing items from the tail
        padding = repeat(padding, times)[-missing:]

    space = index % (len(signal) + 1)
    return np.concatenate((signal[:space], padding, signal[space:]))


def pad_left(signal, padding, minlength):
    """
    Pad a signal with padding value(s) on the start (left side) to be at least the given length.
    """
    return pad_minlength(signal, padding, minlength, 0, fromleft=True)


def pad_right(signal, padding, minlength):
    """
    Pad a signal with padding value(s) on the end to be at least the given length.
    """
    return pad_minlength(signal, padding, minlength, -1, fromleft=False)


def overlap(signal, n):
    """
    Split the 1-d signal into n overlapping parts.
    """
    return np.array([signal[p : len(signal) - q] for p, q in enumerate(reversed(xrange(n)))])


def distances(signal):
    """
    Get the absolute distances from consecutive samples of the signal.
    Signal must have at least two samples.

    See also: np.diff()
    """
    if hasattr(signal, '__len__') and (len(signal) >= 2):
        return np.abs(signal[1:] - signal[:-1])
    else:
        raise ValueError("Expected signal to have at least two samples. Signal was: %s" % signal)


def diffs(signal, start=0, end=0):
    """
    The same as np.ediff1d(). Get the difference of the consecutive samples in signal.
    This differs from distances(), in that the signal is padded with start and end values.

    Signal must have at least two samples.

    See also: distances()
    """
    # Could use np.apply_over_axes - profile with time?
    return np.append(start, signal[1:]) - np.append(signal[:-1], end)


def get_points(signal, size=1000, dtype=np.float64):
    """
    Get coordinate points from a complex signal.
    """
    return complex_as_reals(scale(np.atleast_1d(signal), size), dtype)


def get_pixels(signal, size):
    """
    Get pixel coordinates and pixel values from complex signal on some size.
    """
    # Change bottom-left coordinates to top-left with flip_vertical
    points = get_points(flip_vertical(signal), size) - 0.5

    # Note! np.fix rounds towards zero (rint would round to closest int)
    pixels = np.fix(points).astype(np.int32)
    values = 1 - ((points - pixels) % 1)

    return pixels, values


def scale(signal, size):
    """
    Scale complex signal in unit rectangle area to size and interpret values as pixel centers.
    Range of the coordinates will be from 0.5 to size - 0.5.
    """
    # TODO: Move to math or dsp module
    return ((clip(signal) + 1 + 1j) / 2.0 * (size - 1) + (0.5 + 0.5j))


def scaleto(arr, magnitude=1, inplace=False):
    """
    Scales a complex or real signal by translating startpoint to the origo,
    and scales endpoint to the desired magnitude.
    """
    if inplace:
        arr -= arr[0]
        arr *= magnitude / np.abs(arr[-1])
        return arr
    else:
        return ((arr - arr[0]) * (magnitude / np.abs(arr[-1] - arr[0])))


def abspowersign(base, exponent):
    return np.sign(base) * np.abs(base) ** exponent


def abslogsign(x):
    return np.sign(x) * np.log(np.abs(x))


def lambertw(z):
    """
    Lambert W function:
    http://en.wikipedia.org/wiki/Lambert_W_function
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html
    """
    limit = -1 / np.e
    if limit < np.abs(z) < 0:
        branch = -1
    elif limit < np.abs(z):
        branch = 0
    else:
        raise ValueError("Argument for lambertw %s should be below %s to get real values out." % (z, limit))
    return sc.special.lambertw(z, branch)


def flip_vertical(signal):
    """
    Flip signal on vertical (imaginary) axis.
    """
    return np.array(signal.real - signal.imag * 1j, dtype=np.complex128)


def get_zerocrossings(signal):
    """
    Get the signal transformed so it is one where the signal crosses zero level,
    and zero everywhere else.

    In other words:

    The result will have 1 where the signal crosses the x-axis from
    positive to negative values or vice versa, and 0 elsewhere.
    """
    peaks = pad(distances(np.angle(signal) / np.pi), 0)
    res = np.round((peaks - peaks[1]) * 1.1) * np.sign(np.angle(signal))
    return res


def get_impulses(signal, tau=False):
    """
    Can be used to find where the angle of complex signal crosses zero angle.
    """
    if tau:
        peaks = pad(distances(np.angle(signal) / np.pi % 2), 0)
        res = np.fmax(np.sign((peaks - 1)) * 2, 0)
    else:
        peaks = pad(distances(np.angle(signal) % pi2), 0)
        res = np.fmax(np.sign((peaks - np.pi)) * pi2, 0)
    return res
