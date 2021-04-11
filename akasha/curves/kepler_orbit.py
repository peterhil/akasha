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

from akasha.curves.curve import Curve
from akasha.math import pi2


class KeplerOrbit(Curve):
    """
    Elliptical orbit following Kepler's laws of planetary motion:
    https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time

    Parameters:
    - mu:
      Standard gravitational parameter μ of a celestial body is
      the product of the gravitational constant G and the mass M of
      the body.
    - period:
      Orbital period is the time an astronomical
      object takes to complete one orbit around another object.
    - ecc:
      Orbital eccentricity of an astronomical object determines the
      amount by which its orbit around another body deviates from a
      perfect circle.  A value of 0 is a circular orbit, values
      between 0 and 1 form an elliptic orbit, 1 is a parabolic escape
      orbit, and greater than 1 is a hyperbola.
    - scale:
      Scaling factor for making long periods of time approachable to listening
    """
    def __init__(self, perihelion, semimajor, eccentricity=0.75, period=1, scale=43200):
        # self.mu = mu
        self.perihelion = perihelion  # Distance from sun (origin) to perihelion (closest) point
        self.a = semimajor
        self.period = period
        self.eccentricity = eccentricity
        self.scale = scale

    @property
    def mean_motion(self):
        """
        Mean motion
        https://en.wikipedia.org/wiki/Mean_motion
        """
        return pi2 / self.period

    def mean_anomaly(self, t, ph=0):
        """
        Mean anomaly
        https://en.wikipedia.org/wiki/Mean_anomaly

        ph: time of perihelion passage
        """
        return self.mean_motion * (t - ph)

    def true_anomaly(self, t, ph=0):
        """
        Mean anomaly
        https://en.wikipedia.org/wiki/Mean_anomaly

        ph: time of perihelion passage
        """
        ma = self.mean_anomaly(t, ph=0)
        ecc = self.eccentricity

        # Approximation through Fourier expansion, fails on large eccentricity!
        ta = ma + \
             (2 * ecc - (1 / 4) * ecc ** 3) * np.sin(ma) + \
             (5 / 4) * ecc ** 2 * np.sin(2 * ma) + \
             (13 / 12) * ecc ** 3 * np.sin(3 * ma)

        return ta

    def at(self, t):
        ta = self.true_anomaly(t)
        ecc = self.eccentricity

        # Polar radius formula from focal point for ellipse:
        # https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_focus
        radius = self.a * (1 - ecc ** 2) / (1 + ecc * np.cos(ta))

        return np.array(radius * np.exp(1j * ta), dtype=np.complex128)
