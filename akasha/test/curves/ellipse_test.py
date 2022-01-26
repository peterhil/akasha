#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101

"""
Unit tests for Ellipse
"""

from __future__ import division

import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal

from akasha.curves import Ellipse
from akasha.math import pi2


def ellipse_parameters(ellipse):
    return np.array([ellipse.a, ellipse.b, ellipse.angle, ellipse.origin])


def axis_orientations(a=2.0, b=1.0):
    return np.array([
        [ 1,  1],
        [-1,  1],
        [-1, -1],
        [ 1, -1],
    ]) * (a, b)


half = 1 / 2 * pi2
sixth = 1 / 6 * pi2


class TestEllipse():

    @pytest.mark.parametrize(('params', 'expected'), [
        [( 2,  1, sixth), ( 2,  1, sixth)],
        [(-2,  1, sixth), ( 2,  1, sixth + half)],
        [(-2, -1, sixth), ( 2,  1, sixth + half)],
        [( 2, -1, sixth), ( 2,  1, sixth)],
        ])
    def test_init_normalisation(self, params, expected):
        ell = Ellipse(*params)
        assert_array_almost_equal(ellipse_parameters(ell)[:3], expected)

    @pytest.mark.parametrize(('params', 'expected'), [
        # B as semi-major axis
        [( 1,  2, sixth), ( 1,  2, sixth)],
        [(-1, -2, sixth), ( 1,  2, sixth + half)],
        ])
    def test_init_normalisation_b_major(self, params, expected):
        ell = Ellipse(*params)
        assert_array_almost_equal(ellipse_parameters(ell)[:3], expected)

    @pytest.mark.parametrize(('angle'), [
        half,
        -sixth,
        -3 * half,
        ])
    def test_init_angle(self, angle):
        ell = Ellipse(2, 1, angle)
        assert ell.angle == (angle % pi2), 'Ellipse angle should be modulo pi2!'

    @pytest.mark.parametrize(('ab', 'majmin'), [
        [( 2,  1), (2, 1)],
        [( 3,  4), (4, 3)],
        [(-6,  5), (6, 5)],
        [(-7, -8), (8, 7)]
        ])
    def test_major_minor_axes(self, ab, majmin):
        ell = Ellipse(*ab)
        assert ell.major > ell.minor > 0, 'The semi-major axis should be greater than the semi-minor axis!'
        assert (ell.major, ell.minor) == majmin

    def test_at(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_array_almost_equal(
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
        assert_array_almost_equal(
            np.array([
                2.0006041824631042,  0.9778297511146608,  0.7341007514023552,  1.5443093080698087,
                1.5443093080698089,  0.7341007514023554,  0.9778297511146605,  2.0006041824631042
            ]),
            ell.curvature(np.linspace(0, 1, 8))
        )

    def test_curvature_with_angle(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_array_almost_equal(
            np.array([
                1.0886620913508018,  0.7137017414039344,  1.3762738179669729,  1.7126777269878259,
                0.7690728693063869,  0.8889215911627909,  1.9635436556069104,  1.0886620913508018
            ]),
            ell.curvature(np.linspace(0, 1, 8))
        )

    def test_roc(self):
        ell = Ellipse(1, 0.707, 0)
        assert_array_almost_equal(
            np.array([
                0.4998489999999999,  1.0226729130097203,  1.3622108383484099,  0.6475386729682239,
                0.6475386729682238,  1.3622108383484095,  1.0226729130097205,  0.4998489999999999
            ]),
            ell.roc(np.linspace(0, 1, 8))
        )

    def test_roc_with_angle(self):
        ell = Ellipse(1, 0.707, pi2 * 1/8)
        assert_array_almost_equal(
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
        assert_array_almost_equal(
            np.array([exp.a, exp.b, exp.angle, exp.origin]),
            np.array([ell.a, ell.b, ell.angle, ell.origin])
        )

    @pytest.mark.xfail()
    @pytest.mark.parametrize(('ellipse', 'coefficients'), [
        # https://math.stackexchange.com/questions/993625/ellipse-3x2-x6xy-3y5y2-0-what-are-the-semi-major-and-semi-minor-axes-dis
        (
            Ellipse(0.514256, 0.375, angle=-1.353736, origin=-0.207206-0.156022j),
            np.array([0.258716, 0.052086, 0.146368, 0.115342, 0.056466, -0.020835])
        ),
        (
            Ellipse(
                np.sqrt(7.0 / (12.0 * (4.0 + np.sqrt(10)))),
                np.sqrt(7.0 / (12.0 * (4.0 - np.sqrt(10)))),
                angle=-np.arctan((np.sqrt(10.0) - 1.0) / 3.0),
                origin= (-1.0 / 3.0) + 0.5j
            ),
            np.array([3, 6, 5, -1, -3, 0], dtype=np.float64)
        ),
        ])
    def test_general_coefficients(self, ellipse, coefficients):
        # coefficients = np.array([ 0.258716,  0.052086,  0.146368,  0.115342,  0.056466, -0.020835])
        # ellipse = Ellipse(0.514256, 0.375, angle=-1.353736, origin=-0.207206-0.156022j)
        assert_array_almost_equal(
            ellipse.general_coefficients,
            coefficients
        )

    @pytest.mark.xfail()
    def test_ellipse_from_general_coefficients(self):
        coefficients = np.array([ 0.445 , -0.39  ,  0.445 ,  0.036 ,  0.164 , -0.1368])
        ellipse = Ellipse.from_general_coefficients(*coefficients)
        original = Ellipse(0.8, 0.5, -0.375*pi2, -0.15-0.25j)

        # assert_array_almost_equal(
        #     ellipse.general_coefficients,
        #     coefficients,
        #     err_msg='Ellipse from coefficients do not match the coefficients'
        # )
        assert_array_almost_equal(
            ellipse_parameters(ellipse),
            ellipse_parameters(original),
            err_msg='Ellipse does not match the original'
        )

    @pytest.mark.xfail()
    def test_general_coefficients_cycle(self):
        ell = Ellipse(0.514256, 0.375, angle=-1.353736, origin=-0.207206-0.156022j)
        assert_array_almost_equal(
            ell.general_coefficients,
            Ellipse.from_general_coefficients(*ell.general_coefficients).general_coefficients
        )
