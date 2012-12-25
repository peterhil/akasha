#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for time
"""

import pytest
import timeit

from akasha.audio.time import Chrono, ps, ns, us, ms, seconds, minutes, hours, days, months, years
from akasha.types.numeric import RealUnit
from cdecimal import Decimal, getcontext


class TestChrono(object):
    """
    Tests of time
    """
    @classmethod
    def setup_class(cls):
        assert issubclass(Chrono, RealUnit)
        assert issubclass(Chrono, object)

    def test_init(self):
        """It should intialize"""
        # pylint: disable=W0212
        t = 0.1
        c = Chrono(t)
        assert isinstance(c, Chrono)
        assert isinstance(c._sec, Decimal)
        assert t == c

    def test_precision(self):
        ctx = getcontext()
        assert 32 == ctx.prec

    @pytest.mark.parametrize(('prefix', 'factor'), [
        [ps, 1e-12],
        [ns, 1e-9],
        [us, 1e-6],
        [ms, 1e-3],
        [seconds, 1],
        [minutes, 60],
        [hours, 3600],
        [days, 86400],
        [months, 27.321661569284 * 86400],
        [years, 365.256363004 * 86400],
    ])
    def test_prefix(self, prefix, factor):
        value = 500
        assert (value * factor) == prefix(value) == getattr(Chrono, prefix.__name__)(value)

    def test_now(self):
        latency = timeit.timeit(Chrono.now, number=3)
        diff = -(Chrono.now() - Chrono.now())
        assert 0 < diff < latency
