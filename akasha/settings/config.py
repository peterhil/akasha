#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Akasha configuration module.
"""

import logging

from fractions import Fraction


__all__ = ['config']


class sampling(object):
    FRAMETIME = 40  # 1000 / 25 fps
    RATE = 44100
    ANTIALIAS = False
    NEGATIVE = False


class audio(object):
    BUFFERSIZE = 512
    CHANNELS = 1
    SAMPLETYPE = -16


class frequency(object):
    BASE = 54.0  # 432 Hz / 8
    AUDIBLE_MIN = 1.0
    AUDIBLE_MAX = 22_000.0
    RATIO_MIN = Fraction(1, sampling.RATE)
    RATIO_MAX = Fraction(1, 2)


class logging_limits(object):
    """
    Various limits for logging
    """
    LOGLEVEL = logging.INFO
    FREQUENCY_DEVIATION_CENTS = 0.1
    LOOP_THRESHOLD_PERCENT = 90


class config(object):
    """
    Configuration class.
    """
    audio = audio
    frequency = frequency
    sampling = sampling
    logging_limits = logging_limits
