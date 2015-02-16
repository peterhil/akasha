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


def z_transform(signal, m=None, w=None, a=1.0, dtype=np.complex128):
    """
    Chirp Z-transform as described in "The Chirp z-Transform Algorithm" by
    L.R. Rabiner, R.W. Schafer and C.M. Rader in the Bell System Technical Journal, May 1969.
    http://cronos.rutgers.edu/~lrr/Reprints/015_czt.pdf
    """
    signal = dtype(signal)
    l = len(signal)
    if m is None: m = l
    if w is None: w = np.exp(-1j * pi2 / m)
    z = np.zeros(m, dtype=dtype)
    n = np.arange(l)
    k = np.arange(m)
    # 1: Next largest power of two for m + l -1
    lp2 = power_limit(m + l - 1, 2, np.ceil)
    # 2: y[n] = A**-n*W**(n**2)/2 * x[n], zero padded on right to length lp2
    def chirp(x):
        return w ** ((x ** 2) / 2.0)
    # chirp = w ** (np.arange(1 - x, max(m, x)) ** 2 / 2.0)
    print('chirp', chirp(n))
    y = np.append(signal * (a ** -n) * chirp(n), np.zeros(lp2 - l))  # Zero pad to lp2 length
    print('y', y, len(y))
    # 4: Vn =
    # # pc = np.zeros(lp2, dtype=dtype)
    # chirp2 = lambda n: (w ** ((-n ** 2) / 2.0))
    # # chirp3 = lambda n: (w ** ((-(lp2 - n) ** 2) / 2.0))
    # # pc[0:m - 1] = chirp2(n[:m - 1])
    # # pc[lp2 - l + 1:lp2] = chirp3(np.arange(lp2 - l + 1, lp2))
    # pc = pad(chirp2(np.arange(lp2 - l + 1, m + 1)), count=(lp2 - l), value=0.0)
    # print('pc', pc, len(pc))
    # vn = np.roll(pc, int(lp2 - l + 1))  # Double check this roll
    # # vn = pc
    @np.vectorize
    def v(n):
        if 0 <= n <= m - 1:
            return (w ** ((-n ** 2) / 2.0))
        if lp2 - l + 1 <= n < lp2:
            return (w ** ((-(lp2 - n) ** 2) / 2.0))
        else:
            return 0
    vn = v(np.arange(lp2))
    print('vn', vn)
    # 3, 5, 6, 7:
    g = sc.ifft(sc.fft(y) * sc.fft(vn))[:m]
    print('g', g)
    # 8:
    wk = (w ** ((k ** 2) / 2.0))
    return (g * wk)


def z_transform_naive(signal, m=None, w=None, a=1.0, dtype=np.complex128):
    """
    Naive and slow O(n**2) implementation of Rader's chirp z-transform. For testing, do not use!
    """
    signal = dtype(signal)
    l = len(signal)
    if m is None: m = l
    if w is None: w = np.exp(-1j * pi2 / m)
    z = np.zeros(m, dtype=dtype)

    for k in range(m):
        for n in range(l):
            # print("l = {}, m = {}, k = {}, n = {}, w = {}, a = {}, z = {}".format(l, m, k, n, w, a, z))
            z[k] += signal[n] * a * w ** (k * n)
    return z
