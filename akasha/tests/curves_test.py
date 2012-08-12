#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for curves.py
"""

import pytest
import numpy as np

from numpy.testing.utils import assert_array_almost_equal, assert_array_max_ulp
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff
from numpy.testing.utils import nulp_diff

from numbers import Number

from .. import akasha

from akasha.audio.curves import Circle, Curve, Super
from akasha.utils.math import pi2


class TestCurve(object):

    def test_call(self):
        c = Curve()
        with pytest.raises(NotImplementedError):
            c(4)

    def test_at(self):
        c = Curve()
        assert hasattr(c, 'at')
        with pytest.raises(NotImplementedError):
            c.at(4)


class TestCircle(object):

    def test_at(self):
        c = Circle()
        assert_nulp_diff(c.at(np.arange(-1, 3)), 1+0j, 3)

        pts = np.linspace(0, 1.0, 7, endpoint=False)
        assert_nulp_diff(c.at(pts), np.exp(pi2 * 1j * pts), 1)


class TestSuper(object):

    def test_init(self):
        s = Super()
        assert s.superness == (4, 2, 2, 2, 1.0, 1.0)

    superness_params = [
        [(),                    (4, 2, 2, 2, 1.0, 1.0)],
        [None,                  (4, 2, 2, 2, 1.0, 1.0)],
        [7,                     (7, 7, 7, 7, 1.0, 1.0)],
        [[4],                   (4, 4, 4, 4, 1.0, 1.0)],
        [[3, 1],                (3, 1, 1, 1, 1.0, 1.0)],
        [(4, 4, 2),             (4, 4, 2, 2, 1.0, 1.0)],
        [(4, 3.1, 2.2, 1.3),    (4, 3.1, 2.2, 1.3, 1.0, 1.0)],
        [[5, 1, 2, 3, 0.5],     (5, 1, 2, 3, 0.5, 1.0)],
    ]

    @pytest.mark.parametrize(("inp", "out"), superness_params)
    def test_normalise_superness(self, inp, out):
        assert Super.normalise_superness(inp) == out

    @pytest.mark.parametrize(("inp", "out"), superness_params)
    def test_normalise_superness_from_init(self, inp, out):
        assert Super(inp).superness == out

