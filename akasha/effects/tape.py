#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2011 Peter Hillerström. All rights reserved.
#
# Author: Peter Hillerström
# Date: 2011-12-11
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Tape compression module.
"""

import cmath
import funcy
import numpy as np

from builtins import range

from akasha.utils.log import logger, logging
from akasha.math import map_array, normalize, as_polar, as_rect


class Magnet():
    """Model magnetization direction of particles.

    Vary m between 0 and 1, and return signal (s) with value
    between -1 and 1 depending on the current magnetization (m) level.
    """
    def __init__(self, m=0.5):
        self.m = np.clip(m, 0, 1)

    def hysteresis(self, s):
        m = self.m
        if s > 0:
            m = np.clip(m + (1 - m) * s, 0, 1)
        else:
            m = np.clip(m + m * s, 0, 1)
        return m

    def demagnetize(self, m=0.5):
        self.m = m
    def __call__(self, s):
        m = self.m = self.hysteresis(s)
        s = (m * 2 - 1)
        return s


def complex_magnet(signal, gain=1.0):
    m = Magnet()
    mv = np.vectorize(m)
    s = np.asarray(signal) * gain
    return mv(np.ediff1d(s.real)) + 1j * mv(np.ediff1d(s.imag))


def magnetize(x0, x1, m, norm_level=0.95):
    """Get previous magnetization (m) level and diff (x) in signal
    level in. Return new magnetization level.

    Should be: Get two input samples in and compare their
    level and difference to the current magnetization level
    to get the change in output signal level.
    """
    # d_in = x1 - x0   # Can be at most (+-)2 (from -1 to +1)
    d_out = x0 / norm_level
    # # Prevent zero division
    # d_out /=  min((x0 / d_in), norm_level)

    # Remaining polarisation suspectibility:
    # From 0 to norm_level (if m <= norm_level)
    perm = (np.sign(m) * 1.0) - m

    # logger.log(
    #     logging.BORING, "Delta in: %s, out: %s, Permeability: %s",
    #     d_in, d_out, perm
    # )

    return m + perm * d_out


def mag2(x0, x1, m, norm_level=0.95):
    """Get previous magnetization (m) level and diff (x) in signal level in.
    Return new magnetization level.
    """
    d_in = x1 - x0

    permeability = (norm_level - m)
    if np.sign(d_in) == -1:
        permeability = 1 - permeability
    # Should d_in be abs(d_in)?
    d_out = permeability * d_in / norm_level

    # msg = "Delta in: %s, Permeability: %s, Change: %s"
    # logger.log(logging.BORING, msg % (d_in, permeability, d_out))

    return m + d_out


def mag(x, m, norm_level=1.0):
    """Get previous magnetization (m) level and diff (x) in signal level in.
    Return new magnetization level.
    """
    r = m + (x * norm_level - x * abs(m))
    # normalize to prevent oscillation
    r = min(np.abs(r), norm_level) * np.sign(r)

    return r


def tape_compress(signal, norm_level=0.95):
    """Model tape compression hysteresis.
    """
    if (signal[0] == complex):
        amp = np.abs(signal)
    else:
        amp = signal
    # diff_in = np.abs(np.ediff1d(amp))
    # Calculate result - could use np.ufunc.accumulate?
    out = np.empty(len(amp))
    out[0] = signal[0]
    for i in range(len(amp) - 1):
        out[i + 1] = mag2(amp[i], amp[i + 1], out[i], norm_level)
    return out


def cx_tape_compress(signal, norm_level=0.95):
    """Model tape compression hysteresis on a complex analytic signal.
    """
    angles = np.exp(np.angle(signal) * 1j)
    compressed = tape_compress(np.abs(signal), norm_level)

    return compressed * angles


def gamma(g, amp, signal):
    """Calculate the gamma curve.
    """
    return signal ** g * amp


def gamma_compress(signal, g, amp=1.0, normal=True):
    """Apply gamma compression on the amplitude of a complex signal.

    Example usage
    -------------
    dur = 30
    eine = normalize(read("Amadeus - Eine Kleine.aiff", fs=44100, dur=dur))
    compressed = gamma_compress(eine[:dur * 44100], 0.487, 105)
    anim(Pcm(normalize(compressed[:dur * 44100]) * 0.5, 1), antialias=True)
    """
    assert amp != 0, "Amplitude can not be zero!"
    phi = as_polar(signal)
    if normal:
        phi[:, 0] = normalize(phi[:, 0])

    vgamma = np.vectorize(funcy.curry(gamma, 3)(g)(amp))
    phi[:, 0] = np.apply_along_axis(vgamma, 0, phi[:, 0]) * (1.0 / amp)

    return as_rect(phi).T
