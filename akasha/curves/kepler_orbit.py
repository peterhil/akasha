#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Ellipses module
"""

from __future__ import division

import numpy as np
from scipy.optimize import newton

from akasha.curves.curve import Curve
from akasha.math import pi2
from akasha.utils.python import class_name


class KeplerOrbit(Curve):
    """
    Elliptical orbit following Kepler's laws of planetary motion:
    https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time

    Parameters:
    - period:
      Orbital period is the time an astronomical
      object takes to complete one orbit around another object.
    - ecc:
      Orbital eccentricity of an astronomical object determines the
      amount by which its orbit around another body deviates from a
      perfect circle.  A value of 0 is a circular orbit, values
      between 0 and 1 form an elliptic orbit, 1 is a parabolic escape
      orbit, and greater than 1 is a hyperbola.
    """
    def __init__(
        self,
        perihelion,
        semimajor,
        eccentricity=0.85,
        period=1,
        newton=True,
        name='',
        ph=0
    ):
        self.name = name
        # Distance from sun (origin) to perihelion (closest) point
        self.perihelion = perihelion
        self.a = semimajor
        self.period = period
        self.eccentricity = eccentricity
        self.newton = newton
        self.ph = 0  # Time of perihelion passage

    def __repr__(self):
        return f'{class_name(self)}(' + \
            f'{self.perihelion!r}, ' + \
            f'{self.a!r}, ' + \
            f'eccentricity={self.eccentricity!r}, ' + \
            f'period={self.period!r}, ' + \
            f'name={self.name!r})'

    @property
    def mean_motion(self):
        """
        Mean motion
        https://en.wikipedia.org/wiki/Mean_motion
        """
        return pi2 / self.period

    def mean_anomaly(self, t):
        """
        Mean anomaly
        https://en.wikipedia.org/wiki/Mean_anomaly#Formulae
        """
        # Alternatively use the standard gravitational parameter
        # μ = G * M, see link above for details
        return self.mean_motion * (t - self.ph)

    def eccentric_anomaly(self, t):
        """
        Eccentric anomaly
        https://en.wikipedia.org/wiki/Eccentric_anomaly
        """
        ma = self.mean_anomaly(t)
        ecc = self.eccentricity

        def kepler_equation(e):
            return e - ecc * np.sin(e) - ma

        def ke_prime(e):
            return 1 - ecc * np.cos(e)

        return newton(
            kepler_equation, ma, ke_prime,
            tol=1.48e-08, maxiter=50
        )

    def true_anomaly(self, t):
        """
        True anomaly
        https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly
        """
        ea = self.eccentric_anomaly(t)
        ecc = self.eccentricity
        nu = np.arctan2(
            np.sqrt(1 - ecc ** 2) * np.sin(ea),
            np.cos(ea) - ecc,
        )
        return nu

    def true_anomaly_fourier(self, t):
        """
        True anomaly
        https://en.wikipedia.org/wiki/True_anomaly#From_the_mean_anomaly
        """
        ma = self.mean_anomaly(t)
        ecc = self.eccentricity

        # Approximation through Fourier expansion, fails on large eccentricity!
        ta = ma + \
             (2 * ecc - (1 / 4) * ecc ** 3) * np.sin(ma) + \
             (5 / 4) * ecc ** 2 * np.sin(2 * ma) + \
             (13 / 12) * ecc ** 3 * np.sin(3 * ma)

        return ta

    def at(self, t):
        ecc = self.eccentricity

        if self.newton:
            ta = self.true_anomaly(t)
        else:
            ta = self.true_anomaly_fourier(t)
            # Polar radius formula from focal point for ellipse:
            # https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_focus
        radius = self.a * (1 - ecc ** 2) / (1 + ecc * np.cos(ta))

        return np.array(radius * np.exp(1j * ta), dtype=np.complex128)
