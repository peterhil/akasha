#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Akasha configuration module.
"""

import logging

from fractions import Fraction


__all__ = ['config']


class sampling:
    FRAMETIME = 40  # 1000 / 25 fps
    RATE = 44100
    ANTIALIAS = False
    NEGATIVE = False


class audio:
    BUFFERSIZE = 512
    CHANNELS = 1
    SAMPLETYPE = -16


class frequency:
    BASE = 54.0  # 432 Hz / 8
    AUDIBLE_MIN = 1.0
    AUDIBLE_MAX = 22_000.0
    RATIO_MIN = Fraction(1, sampling.RATE)
    RATIO_MAX = Fraction(1, 2)


class logging_limits:
    """
    Various limits for logging
    """

    LOGLEVEL = logging.INFO
    FREQUENCY_DEVIATION_CENTS = 0.1
    LOOP_THRESHOLD_PERCENT = 90


class config:
    """
    Configuration class.
    """

    audio = audio
    frequency = frequency
    sampling = sampling
    logging_limits = logging_limits
