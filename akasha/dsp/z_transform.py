#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Z-transform
"""

from __future__ import division

import numpy as np
import scipy as sc

from scipy import signal as dsp

from akasha.utils.log import logger
from akasha.utils.math import map_array, pi2, pad, power_limit


def z_transform(signal, m=None, w=None, a=1.0):
    """
    Chirp Z-transform as described in "The Chirp z-Transform Algorithm" by
    L.R. Rabiner, R.W. Schafer and C.M. Rader in the Bell System Technical Journal, May 1969.
    http://cronos.rutgers.edu/~lrr/Reprints/015_czt.pdf
    """
    signal = np.atleast_1d(signal).astype(np.complex)
    l = len(signal)
    if m is None: m = l
    if w is None: w = np.exp(-1j * pi2 / m)
    n = np.arange(l)
    k = np.arange(m)
    lp2 = power_limit(m + l - 1, 2, np.ceil)

    y = np.append(signal * (a ** -n) * chirp(w, n), np.zeros(lp2 - l))
    vn = np.roll(ichirp(w, np.arange(lp2) - l), -l)

    g = sc.ifft(sc.fft(y) * sc.fft(vn))[:m]
    return (g * chirp(w, k))


def chirp(w, n):
    return w ** ((n ** 2) / 2.0)


def ichirp(w, n):
    return w ** ((-n ** 2) / 2.0)


def z_transform_naive(signal, m=None, w=None, a=1.0):
    """
    Naive and slow O(n**2) implementation of Rader's chirp z-transform. For testing, do not use!
    """
    signal = np.atleast_1d(signal).astype(np.complex)
    l = len(signal)
    if m is None: m = l
    if w is None: w = np.exp(-1j * pi2 / m)
    z = np.zeros(m, dtype=np.complex)

    for k in range(m):
        for n in range(l):
            # print("l = {}, m = {}, k = {}, n = {}, w = {}, a = {}, z = {}".format(l, m, k, n, w, a, z))
            z[k] += signal[n] * a * w ** (k * n)
    return z
