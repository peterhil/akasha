#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High precision time module
"""

import sys

if sys.version_info >= (3, 3, 0):
    from decimal import Decimal, getcontext
else:
    from cdecimal import Decimal, getcontext

from timeit import default_timer as clock

from akasha.types.numeric import RealUnit
from akasha.utils.python import _super


getcontext().prec = 32


class Chrono(RealUnit):
    """
    Chrono is a nanosecond precision time class, that is compatible
    with OSC (Open Sound Control).

    Internal format is 64-bit fixed number with 32 bits for seconds and
    another 32 bits for parts of a second.
    """

    def __init__(self, secs):
        _super(self).__init__()
        self._sec = Decimal(float(secs))

    @property
    def _unit(self):
        return '_sec'

    @classmethod
    def now(cls):
        """Current time as unix timestamp.

        Convert to datetime: datetime.fromtimestamp(Chrono.now())
        datetime.datetime(2012, 12, 27, 3, 41, 9, 206083)
        """
        return cls(clock())

    @classmethod
    def add_prefix(cls, factor, name, symbol=None):
        """Add a derived time unit with a name, prefixed symbol
        and a factor to multiply seconds.
        """

        def derived(secs):
            # pylint: disable=C0111
            return cls(secs * factor)

        derived.__doc__ = f'Chrono time as {name}.'
        derived.__name__ = name
        setattr(cls, name, staticmethod(derived))
        if symbol:
            setattr(cls, symbol, staticmethod(derived))
        return derived


ps = picoseconds = Chrono.add_prefix(1e-12, 'picoseconds', symbol='ps')
ns = nanoseconds = Chrono.add_prefix(1e-9, 'nanoseconds', symbol='ns')
us = microseconds = Chrono.add_prefix(1e-6, 'microseconds', symbol='us')
ms = milliseconds = Chrono.add_prefix(1e-3, 'milliseconds', symbol='ms')
sec = seconds = Chrono.add_prefix(1, 'seconds', symbol='sec')
minutes = Chrono.add_prefix(60, 'minutes', symbol='min')
hours = Chrono.add_prefix(3600, 'hours', symbol='h')
days = Chrono.add_prefix(86400, 'days', symbol='d')
weeks = Chrono.add_prefix(7 * 86400, 'weeks', symbol='w')
# sidereal month
months = Chrono.add_prefix(27.321661569284 * 86400, 'months', symbol='m')
# Annum = avg. tropical year
years = Chrono.add_prefix(365.24219265 * 86400, 'years', symbol='a')
