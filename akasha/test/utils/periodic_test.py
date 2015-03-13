#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for periodic arrays
"""

import pytest
import numpy as np

from numpy.testing.utils import assert_array_equal

from akasha.utils.periodic import period


class TestPeriod(object):
    """Test period arrays"""

    @pytest.mark.parametrize(("param", "shape"), [
        [(),                    ()],
        [[],                    ()],
        [None,                  ()],
        [0,                     (0, )],
        [1,                     (1, )],
        [[2, 3],                (2, 3)],
        [[4, 5, 6],             (4, 5, 6)],
    ])
    def test_init(self, param, shape):
        pa = period(param)
        assert isinstance(pa, period)
        assert pa.shape == shape

    @pytest.mark.parametrize(("seq"), [
        [None],
        [()],
        [[]],
        [(1, 2, 3)],
        [u'foo'],
        [np.inf],
        [[2 + 5j, -3 + 6j]],
        [np.array([[4.1, 5.2], [6.3, 7.4]])],
        [np.arange(6)],
    ])
    def test_array(self, seq):
        assert_array_equal(
            np.array(seq),
            period.array(seq)
        )

    def test_index_mod(self):
        # pylint: disable=W0212
        n = 4
        pa = period.array(np.arange(n))
        for i in np.arange(-2, n + 2):
            assert i % n == pa._mod(i)

    def test_index_mod_slice(self):
        ar = np.arange(6).reshape(2, 3)
        pa = period.array(ar)

        assert_array_equal(ar[::], pa[::])
        # assert_array_equal(ar[1, slice(None)], pa[1, slice(None)])

        assert_array_equal(ar, pa[::])
        assert_array_equal(ar, pa[:])
        assert_array_equal(ar[:], pa[:])

        assert_array_equal(ar[1:], pa[1:])

        assert_array_equal(ar[1], pa[1])
        assert_array_equal(ar[0], pa[4])

        # assert_array_equal(ar[0:1, 1:-2], pa[0:1, 1:-2])
        # assert_array_equal(ar[:, 1:-2], pa[:, 1:-2])

    def test_view(self):
        ar = np.arange(6).reshape(2, 3)
        pa = ar.view(period)
        assert np.array_equal(ar, pa)

    def test_item_1d(self):
        pa = period(1)
        pa[2] = 17
        assert_array_equal(np.repeat(17, 3), pa[[1, 2, 3]])

    def test_getitem_1d_wraps(self):
        n = 3
        ind = np.arange(-n - 1, n * 2)
        ar = np.arange(n)
        pa = period(n)
        pa[::] = ar

        assert_array_equal(ar[ind % len(ar)], pa[ind])

    def test_getitem_1d_view(self):
        n = 4
        ind = np.arange(-n - 1, n * 2)
        ar = np.arange(n)
        pa = ar.view(period)

        assert_array_equal(ar[ind % len(ar)], pa[ind])

    def test_getitem_2d(self):
        shape = (2, 3)
        pa = period(shape)
        ar = np.arange(6).reshape(shape)
        pa[::] = ar

        assert_array_equal(ar, pa)
