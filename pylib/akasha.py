#!/usr/bin/env python
# encoding: utf-8
"""
akasha.py

Created by Peter on 2011-12-11.
Copyright (c) 2011 Loihde. All rights reserved.
"""

def setup():
    np.set_printoptions(precision=16, suppress=True)

    # Set the user's default locale, see http:// docs.python.org/library/locale.html
    # Also be sure to have LC_ALL='fi_FI.UTF-8' and CHARSET='UTF-8' set in the environment
    # to have sys.stdin.encoding = UTF-8
    locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')
    assert sys.stdin.encoding == 'UTF-8', "Unicode not enabled! Current input encoding is: %s" % sys.stdin.encoding


if __name__ == '__main__':
    import locale
    import logging
    import sys

    import types
    import numpy as np

    from collections import defaultdict
    from fractions import Fraction
    from cmath import rect, pi, exp, phase
    from numbers import Number
    from scipy.signal import hilbert

    from audio.dtmf import DTMF
    from audio.envelope import Attack, Exponential
    from audio.harmonics import Overtones
    from audio.noise import *
    from audio.oscillator import Osc, Super, Frequency
    from audio.sound import Sound

    from io.audio import play, write, read
    from io.keyboard import *

    from tunings import WickiLayout

    from utils.animation import *
    from utils.graphing import *
    from utils.math import *
    from utils.splines import *
    from utils.log import logger, ansi

    setup()

    def make_test_sound(freq = 230):
        h = Overtones(Osc(freq), damping=lambda f, a=1.0: (-f/100.0, a/(f/freq)), n = 20)
        c = Chaos()
        o2 = Osc(220)
        o4 = Osc(440)
        o3 = Osc(330)
        s = Sound(h, o2, o3, o4)
        return s

