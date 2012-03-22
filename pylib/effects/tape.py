#!/usr/bin/env python
# encoding: utf-8
"""
compression.py

Created by Peter on 2011-12-11.
Copyright (c) 2011 Loihde. All rights reserved.
"""

from __future__ import absolute_import

import numpy as np
import logging

from ..utils.log import logger
from ..utils.math import diffs


def magnetize(x0, x1, m, norm_level=0.95):
    """Get previous magnetization (m) level and diff (x) in signal level in. Return new magnetization level.
    Should be: Get two input samples in and compare their level and difference to
    the current magnetization level to get the change in output signal level.
    """
    d_in = x1 - x0   # Can be at most (+-)2 (from -1 to +1)
    d_out = x0 / norm_level #/ min((x0 / d_in), norm_level) # Prevent zero division
    perm = (np.sign(m) * 1.0) - m   # Remaining polarisation suspectibility: From 0 to norm_level (if m <= norm_level)
    #logger.log(logging.BORING, "Delta in: %s, out: %s, Permeability: %s" % (d_in, d_out, perm))
    return m + perm * d_out

def mag2(x0, x1, m, norm_level=0.95):
    permeability = (norm_level - m)
    d_in = x1 - x0
    # Should d_in be abs(d_in)?
    d_out = permeability * d_in / max(x0, x1, norm_level)
    #logger.log(logging.BORING, "Delta in: %s, Permeability: %s, Change: %s" % (d_in, permeability, d_out))
    return m + d_out

def mag(x, m, norm_level=1.0):
    r = m + (x * norm_level - x * abs(m))
    r = min(np.abs(r), norm_level) * np.sign(r) # normalize to prevent oscillation
    return r

def tape_compress(signal, norm_level=0.95):
    """Model tape compression hysteresis."""
    if (signal[0] == complex):
        amp = np.abs(signal)
    else:
        amp = signal
    #diff_in = np.abs(diffs(amp))
    # Calculate result - could use np.ufunc.accumulate?
    out = np.empty(len(amp))
    out[0] = signal[0]
    for i in xrange(len(amp)-1):
        out[i+1] = mag2(amp[i], amp[i+1], out[i], norm_level)
    return out

def cx_tape_compress(signal, norm_level=0.95):
    return tape_compress(signal, norm_level) * np.exp(np.angle(signal)*1j)
