#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Curve
"""

import pytest

from akasha.audio.generators import PeriodicGenerator
from akasha.curves import Curve
from akasha.utils.patterns import Singleton


class TestCurve():

    def test_super(self):
        assert issubclass(Curve, PeriodicGenerator)
        assert not issubclass(Curve, Singleton)

    def test_at(self):
        with pytest.raises(NotImplementedError):
            Curve.at(4)

    def test_call(self):
        c = Curve()
        with pytest.raises(NotImplementedError):
            c(4)

    def test_repr(self):
        c = Curve()
        assert 'Curve()' == repr(c)

    def test_str(self):
        c = Curve()
        assert c.__class__.__name__ in str(c)
