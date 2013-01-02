#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2011 Peter Hillerström. All rights reserved.
#
# Author: Peter Hillerström
# Date: 2011-12-11

"""
Tape compression module.
"""

import cmath
import numpy as np

from funckit import xoltar as fx
from akasha.utils.math import map_array, normalize


def magnetize(x0, x1, m, norm_level=0.95):
    """
    Get previous magnetization (m) level and diff (x) in signal level in.
    Return new magnetization level.

    Should be: Get two input samples in and compare their level and difference to
    the current magnetization level to get the change in output signal level.
    """
    # d_in = x1 - x0   # Can be at most (+-)2 (from -1 to +1)
    d_out = x0 / norm_level  # / min((x0 / d_in), norm_level) # Prevent zero division

    # Remaining polarisation suspectibility: From 0 to norm_level (if m <= norm_level)
    perm = (np.sign(m) * 1.0) - m

    #logger.log(logging.BORING, "Delta in: %s, out: %s, Permeability: %s" % (d_in, d_out, perm))
    return m + perm * d_out


def mag2(x0, x1, m, norm_level=0.95):
    """
    Get previous magnetization (m) level and diff (x) in signal level in.
    Return new magnetization level.
    """
    permeability = (norm_level - m)
    d_in = x1 - x0
    # Should d_in be abs(d_in)?
    d_out = permeability * d_in / max(x0, x1, norm_level)
    # msg = "Delta in: %s, Permeability: %s, Change: %s"
    # logger.log(logging.BORING, msg % (d_in, permeability, d_out))
    return m + d_out


def mag(x, m, norm_level=1.0):
    """
    Get previous magnetization (m) level and diff (x) in signal level in.
    Return new magnetization level.
    """
    r = m + (x * norm_level - x * abs(m))
    r = min(np.abs(r), norm_level) * np.sign(r)  # normalize to prevent oscillation
    return r


def tape_compress(signal, norm_level=0.95):
    """
    Model tape compression hysteresis.
    """
    if (signal[0] == complex):
        amp = np.abs(signal)
    else:
        amp = signal
    #diff_in = np.abs(diffs(amp))
    # Calculate result - could use np.ufunc.accumulate?
    out = np.empty(len(amp))
    out[0] = signal[0]
    for i in xrange(len(amp) - 1):
        out[i + 1] = mag2(amp[i], amp[i + 1], out[i], norm_level)
    return out


def cx_tape_compress(signal, norm_level=0.95):
    """
    Model tape compression hysteresis on a complex analytic signal.
    """
    return tape_compress(signal, norm_level) * np.exp(np.angle(signal) * 1j)


# 10.9.2012

def as_polar(signal):
    """
    Return a complex signal in polar coordinates.
    """
    return np.array([np.abs(signal), np.angle(signal)]).T


def gamma(g, amp, signal):
    """
    Calculate the gamma curve.
    """
    return signal ** g * amp


def gamma_compress(signal, g, amp=1.0, normal=True):
    """
    Apply gamma compression on the amplitude of a complex signal.
    """
    phi = as_polar(signal)
    if normal:
        phi[:, 0] = normalize(phi[:, 0])

    vgamma = np.vectorize(fx.curry(gamma, g, amp))
    phi[:, 0] = np.apply_along_axis(vgamma, 0, phi[:, 0]) * (1.0 / amp)  # @TODO amp can't be zero

    return np.array(map_array(lambda x: cmath.rect(*x), phi)).T

# eine = normalize(read("Amadeus - Eine Kleine.aiff", fs=44100, dur=6*60+32))
# gr082 = gamma_compress(eine, 0.81968, 105)
# anim(Pcm(gr082), antialias=True, dur=60*6+30)
