#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Z-transform
"""


import numpy as np
import scipy as sc
import sys

if sys.version_info >= (3, 5):
    from scipy.fft import fft, ifft
else:
    [fft, ifft] = sc.fft, sc.ifft

from akasha.math import pi2, power_limit


__all__ = ['czt', 'iczt']


def czt(signal, m=None, w=None, a=1.0, normalize=False):
    """
    The Chirp Z-transform.

    Transforms the time domain signal into complex frequency domain.

    The chirp Z-transform is implemented as described in
    "The Chirp z-Transform Algorithm" by L.R. Rabiner, R.W. Schafer and
    C.M. Rader in the Bell System Technical Journal, May 1969.

    http://cronos.rutgers.edu/~lrr/Reprints/015_czt.pdf
    """
    signal = np.atleast_1d(signal).astype(np.complex128)
    length = len(signal)
    if m is None:
        m = length
    if w is None:
        w = np.exp(-1j * pi2 / m)
    n = np.arange(length)
    k = np.arange(m)
    lp2 = int(power_limit(m + length - 1, 2, np.ceil))

    y = signal * (a ** -n) * chirp(w, n)
    y_pad = np.append(y, np.zeros(lp2 - length))  # zero pad
    vn = np.roll(ichirp(w, np.arange(lp2) - length), -length)

    g = ifft(fft(y_pad) * fft(vn))[:m]
    scale = 1.0 / length if normalize else 1.0
    return scale * (g * chirp(w, k))


def iczt(signal, m=None, w=None, a=1.0, normalize=False):
    """
    The inverse chirp Z-transform

    Transforms the complex frequency domain signal back to
    time domain signal.

    Uses the conjugation property of Z-transforms to get the inverse.
    """
    signal = np.atleast_1d(signal).astype(np.complex128)
    length = len(signal)
    if normalize:
        signal *= length ** 2
    if m is None:
        m = length
    if w is None:
        w = np.exp(-1j * pi2 / m)

    return np.conjugate(czt(np.conjugate(signal), m, w, a)) / m


def chirp(w, n):
    return w ** ((n ** 2) / 2.0)


def ichirp(w, n):
    return w ** ((-(n ** 2)) / 2.0)


def czt_naive(signal, m=None, w=None, a=1.0):
    """Naive and slow O(n**2) implementation of Rader's chirp
    z-transform. For testing, do not use!
    """
    signal = np.atleast_1d(signal).astype(np.complex128)
    length = len(signal)
    if m is None:
        m = length
    if w is None:
        w = np.exp(-1j * pi2 / m)
    z = np.zeros(m, dtype=np.complex128)

    for k in range(m):
        for n in range(length):
            z[k] += signal[n] * a ** -n * w ** (k * n)
    return z
