#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for z-transform
"""

import pytest
import numpy as np

from numpy.testing.utils import assert_array_equal, assert_array_almost_equal, assert_array_almost_equal_nulp

from akasha.dsp.z_transform import *


z_transforms = [
    [z_transform_naive],
    [z_transform]
    ]


class TestZTransform(object):
    """Test z-transform."""

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_scalar(self, Z):
        assert_array_almost_equal(Z(1), np.ones(1))

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_float(self, Z):
        assert_array_almost_equal(
            Z([1.5, 4, 5], w=0.25),
            np.array([ 10.5, 2.8125, 1.76953125])
            )

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_impulse(self, Z):
        assert_array_almost_equal(
            Z([1, 0, 0]),
            np.ones(3)
            )

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_m(self, Z):
        assert_array_almost_equal(
            Z(1, m=3),
            np.ones(3)
            )

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_a(self, Z):
        assert_array_almost_equal(
            Z([0, 1j, -1, -1j], a=0.5),
            np.array([-4-6j, 14, -4+6j, -6])
            )

    @pytest.mark.parametrize(('Z',), z_transforms)
    def test_float_input_gives_complex_results(self, Z):
        assert_array_almost_equal(
            Z(np.array([1.0, 2.0, 3.0])),
            np.array([6.0, -1.5+0.8660254037844379j, -1.5-0.8660254037844417j])
            )
