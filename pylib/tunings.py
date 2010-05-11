#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction


def cents(*args):
    """Calculate cents from interval or frequency ratio(s).
    When using frequencies, give greater frequencies first."""
    return 1200 * np.log2(args)

def interval(*cnt):
    """Calculate interval ratio from cents."""
    return np.power(2, cnt) /1200.0

def freq_plus_cents(f, cnt):
    """Calculate freq1 + cents = freq2"""
    return f * interval(cnt)

class RegularTuning:
    def init(self, generators):
        pass

# In [128]: sorted(Fraction(3,2) ** np.array(map(Fraction.from_float, xrange(28))) % Fraction(3,2) + Fraction(1))
# Out[128]: 
# [1.0,
#  1.09375,
#  1.12890625,
#  1.1682940423488617,
#  1.2524410635232925,
#  1.375,
#  1.3850951194763184,
#  1.3918800354003906,
#  1.5625,
#  1.5859375,
#  1.6650390625,
#  1.746337890625,
#  1.75,
#  1.890625,
#  1.92926025390625,
#  1.943359375,
#  1.99755859375,
#  2.0,
# ]
