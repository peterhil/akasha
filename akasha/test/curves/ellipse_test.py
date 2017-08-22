#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

import pytest
import numpy as np

from numpy.testing.utils import assert_array_almost_equal
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.curves import Ellipse
from akasha.curves.ellipse import ellipse_fit_fitzgibbon
from akasha.math import pi2


def ellipse_parameters(ellipse):
    return np.array([ellipse.a, ellipse.b, ellipse.angle, ellipse.origin])


class TestEllipse(object):
    """
    Unit tests for Ellipse
    """

    def test_at(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_nulp_diff(
            np.array([
                 0.7071067811865476+0.7071067811865475j,
                -0.0881957403979775+0.7827588458787578j,
                -0.6061872590833505+0.3808926997730741j,
                -0.8658500540960117-0.3029740376569993j,
                -0.3029740376569996-0.8658500540960116j,
                 0.3808926997730734-0.606187259083351j ,
                 0.7827588458787578-0.0881957403979777j,
                 0.7071067811865477+0.7071067811865474j
            ]),
            ell.at(np.linspace(0, 1, 8))
        )

    def test_curvature(self):
        ell = Ellipse(1, 0.707, 0)
        assert_nulp_diff(
            np.array([
                2.0006041824631042,  0.9778297511146608,  0.7341007514023552,  1.5443093080698087,
                1.5443093080698089,  0.7341007514023554,  0.9778297511146605,  2.0006041824631042
            ]),
            ell.curvature(np.linspace(0, 1, 8))
        )

    def test_curvature_with_angle(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_nulp_diff(
            np.array([
                1.0886620913508018,  0.7137017414039344,  1.3762738179669729,  1.7126777269878259,
                0.7690728693063869,  0.8889215911627909,  1.9635436556069104,  1.0886620913508018
            ]),
            ell.curvature(np.linspace(0, 1, 8))
        )

    def test_roc(self):
        ell = Ellipse(1, 0.707, 0)
        assert_nulp_diff(
            np.array([
                0.4998489999999999,  1.0226729130097203,  1.3622108383484099,  0.6475386729682239,
                0.6475386729682238,  1.3622108383484095,  1.0226729130097205,  0.4998489999999999
            ]),
            ell.roc(np.linspace(0, 1, 8))
        )

    def test_roc_with_angle(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_nulp_diff(
            np.array([
                0.9185586675101447,  1.4011455233847177,  0.7265995959126772,  0.5838810093938404,
                1.3002669056599048,  1.1249586127072337,  0.509283303757721 ,  0.9185586675101447
            ]),
            ell.roc(np.linspace(0, 1, 8))
        )

    def test_form_conjugate_diameters(self):
        para = np.array([
            0.21972613654798550664182243963296+0.05592349761197702023851618946537j,
            0.06351890232019029303156543164732+0.72144561730157619194869766943157j,
            0.64242622053932740833204206865048+0.76634753941247535369285515116644j,
            0.79863345476712266357566250007949+0.10082541972287623055493099855084j
        ])
        ell = Ellipse.from_conjugate_diameters(para)
        exp = Ellipse(
            a = 0.49554415098466492,
            b = 0.39581703245813976,
            angle = 2.1056608982703207,
            origin = (0.43107617854365643+0.41113551851222618j)
        )
        assert_nulp_diff(
            np.array([exp.a, exp.b, exp.angle, exp.origin]),
            np.array([ell.a, ell.b, ell.angle, ell.origin])
        )

    def test_ellipse_fit_fitzgibbon(self):
        points = np.array([
            0.6153730882866389+0.1684349412020227j,
            0.4797595596705834+0.2011243355095417j,
            0.4457112154886673+0.2021464613288909j,
            0.4238023402141161+0.2021545825531006j,
            0.3897696453033182+0.2011425910221709j,
            0.2541605969985578+0.1684310346246555j,
            0.3897512937581410+0.1357296682136179j,
            0.4238205462233292+0.1347250678902793j,
            0.4457088519763241+0.1347276650491862j,
            0.4797772426076383+0.1357426178659449j
        ])
        expected = np.array([
            0.0330831623206757,
            0.0000667165668973,
            0.9462653856399076,
            -0.028776862289463,
            -0.3188002645366254,
            0.0320250431330183
        ])
        # Just check that the result matches Matlab/Octave result
        assert_array_almost_equal(ellipse_fit_fitzgibbon(points), expected)

    samples = np.linspace(0, 1, 17, endpoint=False)
    ellipse_params = [
        [ 0.75, 0.125, {'angle': -0.375 * pi2, 'origin': -0.15-0.25j }],
        # Failure of the Fitzgibbon algorithm - degenerate cases?:
        [ 0.75,  0.25, {'angle': -0.375 * pi2, 'origin': -0.15-0.25j }],
        [ 0.75,  0.35, {'angle':  0.375 * pi2, 'origin': -0.15-0.25j }],
    ]
    @pytest.mark.parametrize(('a', 'b', 'kwargs'), ellipse_params)
    def test_ellipse_from_points(self, a, b, kwargs):
        original = Ellipse(a, b, **kwargs)
        fitted = Ellipse.from_points(original.at(self.samples))
        assert_array_almost_equal(
            ellipse_parameters(fitted),
            ellipse_parameters(original)
        )

    def test_general_coefficients(self):
        original = Ellipse(0.8, 0.5, -0.375*pi2, -0.15-0.25j)
        fitted = Ellipse.from_points(original.at(self.samples))
        assert_array_almost_equal(
            ellipse_parameters(fitted),
            ellipse_parameters(original)
        )
