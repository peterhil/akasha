#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from timeit import default_timer as clock

from akasha.types.numeric import RealUnit
from cdecimal import Decimal, getcontext


getcontext().prec = 32


class Chrono(RealUnit):
    """
    Chrono is a nanosecond precision time class, that is compatible with OSC.

    Internal format is 64-bit fixed number with 32 bits for seconds and
    another 32 bits for parts of a second.
    """
    def __init__(self, seconds):
        super(self.__class__, self).__init__()
        self._unit = '_sec'
        self._sec = Decimal(float(seconds))

    @classmethod
    def now(cls):
        return cls(clock())

    @classmethod
    def prefix(cls, factor, prefix, long_name=None):
        def derived(seconds):
            return cls(seconds * factor)
        derived.__name__ = long_name or prefix
        setattr(cls, prefix, staticmethod(derived))
        if long_name:
            setattr(cls, long_name, staticmethod(derived))
        return derived

ps = picoseconds = Chrono.prefix(1e-12, 'ps', long_name='picoseconds')
ns = nanoseconds = Chrono.prefix(1e-9, 'ns', long_name='nanoseconds')
us = microseconds = Chrono.prefix(1e-6, 'us', long_name='microseconds')
ms = milliseconds = Chrono.prefix(1e-3, 'ms', long_name='milliseconds')
sec = seconds = Chrono.prefix(1, 'sec', long_name='seconds')
minutes = Chrono.prefix(60, 'min', long_name='minutes')
hours = Chrono.prefix(3600, 'h', long_name='hours')
days = Chrono.prefix(86400, 'd', long_name='days')
weeks = Chrono.prefix(7 * 86400, 'w', long_name='weeks')
months = Chrono.prefix(27.321661569284 * 86400, 'm', long_name='months') # sidereal month approx. for 2012
years = Chrono.prefix(365.256363004 * 86400, 'a', long_name='years') # sidereal year

