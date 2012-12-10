#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for time
"""

import numpy as np
import operator
import pytest
import timeit

from akasha.audio.time import Chrono, ps, ns, us, ms, seconds, minutes, hours, days, months, years
from akasha.types.numeric import RealUnit
from cdecimal import Decimal, getcontext
from timeit import default_timer as clock


class TestChrono(object):
    """
    Tests of time
    """
    def setup_class(cls):
        assert issubclass(Chrono, RealUnit)
        assert issubclass(Chrono, object)

    def test_init(self):
        """It should intialize"""
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
        chrono = lambda: Chrono.now()
        latency = timeit.timeit(chrono, number=3)
        diff = -(chrono() - chrono())
        assert 0 < diff < latency
