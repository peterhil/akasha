#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Ellipse fitting functions
"""

import numpy as np
import pytest

from numpy.testing.utils import assert_array_almost_equal

from akasha.curves import Ellipse
from akasha.curves.ellipse_fit import ellipse_fit_fitzgibbon, ellipse_fit_halir
from akasha.timing import sampler


class TestEllipseFit(object):
    para = np.array([ 0.375+0.125j,  0.000+0.5j,  -0.750-0.25j ])
    ell = Ellipse.from_conjugate_diameters(para)
    times = sampler.slice(0, sampler.rate, 4410)
    points = ell.at(times)

    @pytest.mark.xfail(reason="something is wrong with the shape handling")
    def test_ellipse_fit_fitzgibbon(self):
        """
        Just check that the result from Fitzgibbon fitting matches Matlab/Octave result
        """
        expected = np.array([ 0.528516, -0.63422 ,  0.528516,  0.158555, -0.052852, -0.105703])
        assert_array_almost_equal(ellipse_fit_fitzgibbon(self.points), expected)

    def test_ellipse_fit_halir(self):
        """
        Just check that the result from Halir fitting matches Matlab/Octave result
        """
        expected = np.array([-0.539164,  0.646997, -0.539164, -0.161749,  0.053916,  0.107833])
        assert_array_almost_equal(ellipse_fit_halir(self.points), expected)
