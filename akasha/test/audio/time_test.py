#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for time
"""

import numpy as np
import operator
import pytest

from akasha.audio.time import Chrono
from akasha.types.numeric import AlgebraicField
from cdecimal import Decimal, getcontext


class TestChrono(object):
    """
    Tests of time
    """
    def setup_class(cls):
        assert issubclass(Chrono, AlgebraicField)
        assert issubclass(Chrono, object)

    def test_init(self):
        """It should intialize"""
        t = 0.1
        c = Chrono(t)
        assert isinstance(c, Chrono)
        assert isinstance(c.time, Decimal)
        assert t == c

    def test_precision(self):
        ctx = getcontext()
        assert 32 == ctx.prec