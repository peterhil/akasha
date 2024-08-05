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
import scipy as sc

from cmath import rect

from akasha.curves.curve import Curve
from akasha.curves.ellipse_fit import (
    ellipse_fit_fitzgibbon,
    ellipse_fit_halir,
)
from akasha.math import as_complex
from akasha.math.geometry import is_orthogonal, midpoint, rotate_towards
from akasha.math.geometry.affine_transform import AffineTransform
from akasha.math import pi2
from akasha.utils.python import class_name


__all__ = ['Ellipse']


def ellipse_axes_normalised(a, b, angle=0):
    if float(a) < 0:
        a = -a
        angle += np.pi
    if float(b) < 0:
        b = -b

    return a, b, angle % pi2


class Ellipse(Curve):
    """
    Ellipse curve

    Parameters are normalised so that:
    - Parameter `a` always determines the direction
      (given by the `angle` parameter)
    - Both `a` and `b` are >= 0
    - Parameter `b` can be greater than `a`, so use Ellipse#major
      if you need the major axis length
    """

    def __init__(self, a, b, angle=0, origin=0):
        # Note! If thinking of making a always the semi-major axis,
        # the angle needs to be rotated plus or minus 90 degrees or so.
        # It is not as easy as it sounds, and care needs to be taken
        # to make it correct!
        (a, b, angle) = ellipse_axes_normalised(a, b, angle)
        self.a = a
        self.b = b
        self.angle = angle % pi2
        self.origin = origin

    @property
    def major(self):
        a, b = np.abs(self.a), np.abs(self.b)
        return a if a > b else b

    @property
    def minor(self):
        a, b = np.abs(self.a), np.abs(self.b)
        return b if a > b else a

    def __repr__(self):
        return (
            f'{class_name(self)}({self.a}, {self.b}, '
            + f'{self.angle}, {self.origin})'
        )

    def __hash__(self):
        return hash((self.a, self.b, self.angle, self.origin))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            hash(self) == hash(other)
        else:
            return NotImplemented

    def parametric(self, points):
        """
        General parametric form of ellipse curve
        http://en.wikipedia.org/wiki/Ellipse#General_parametric_form
        """
        cos = self.a * np.cos(points)
        sin = self.b * np.sin(points)
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        x = self.origin.real + cos * cos_angle - sin * sin_angle
        y = self.origin.imag + cos * sin_angle + sin * cos_angle

        return as_complex(np.array([np.asanyarray(x), np.asanyarray(y)]))

    def at(self, tau):
        """Polar form of ellipse relative to center, translated
        and rotated to origin and angle.
        https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
        """
        thetas = np.asanyarray(tau) * pi2
        b_cos = self.b * np.cos(thetas)
        a_sin = self.a * np.sin(thetas)

        radius = self.a * self.b / np.sqrt(b_cos ** 2 + a_sin ** 2)
        angles = thetas + self.angle
        signal = radius * np.exp(angles * 1j) + self.origin

        return signal

    def curvature(self, tau):
        """Curvature of an ellipse.
        http://mathworld.wolfram.com/Ellipse.html formula 59
        """
        t = np.asanyarray(tau) * pi2 + self.angle
        b_cos = self.b ** 2 * np.cos(t) ** 2
        a_sin = self.a ** 2 * np.sin(t) ** 2

        return (self.a * self.b) / (b_cos + a_sin) ** (3 / 2)

    def roc(self, tau):
        """Radius of curvature."""
        return 1.0 / self.curvature(tau)

    def arc_length(self, tau):
        """Arc length of the ellipse.
        Formula (4) from:
        http://paulbourke.net/geometry/ellipsecirc/Abbott.pdf
        """
        # TODO is substracting self.angle necessary?
        rad = np.fmod(np.asarray(tau) * pi2 - self.angle, np.pi)
        return self.a * sc.special.ellipeinc(rad, self.eccentricity ** 2)

    @property
    def circumference(self):
        return 4.0 * self.arc_length(0.25)

    @property
    def eccentricity(self):
        """Eccentricity of the ellipse:
        https://en.wikipedia.org/wiki/Ellipse#Eccentricity
        """
        return np.sqrt(1.0 - (self.minor / self.major) ** 2)

    @classmethod
    def from_rhombus(cls, para):
        a, b, c, d = para
        para_origin = para - midpoint(a, c)
        k, m = np.abs(para_origin)[:2]
        return cls(m, k, np.angle(para_origin)[3], midpoint(a, c))

    @classmethod
    def from_parallelogram(cls, para):
        dia = np.array([1, 1j, -1, -1j])
        sq = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        center = midpoint(para[0], para[2])
        para_at_origin = para - center

        tr = AffineTransform()
        tr.estimate(dia, para_at_origin)

        u, s, v = np.linalg.svd(
            tr.params[:2, :2], full_matrices=False, compute_uv=True
        )
        a, b = s[:2]

        uv = np.eye(3)
        uv[:2, :2] = u * np.diag(s) * v
        uv = AffineTransform(uv)

        return cls(
            a,
            b,
            # np.angle(uv(dia[0])),
            # np.angle(tr.inverse(dia))[0],
            # pi2 / 4 + tr.rotation + np.tan(tr.shear),
            np.angle(tr(sq)[1]),
            center,
        )

    @classmethod
    def from_conjugate_diameters(cls, para):
        """Find the major and minor axes of an ellipse from a parallelogram
        determining the conjugate diameters.

        Uses Rytz's construction for algorithm:
        http://de.wikipedia.org/wiki/Rytzsche_Achsenkonstruktion#Konstruktion
        """
        para = np.asarray(para, dtype=np.complex128)
        c = midpoint(para[0], para[2])
        para = para - c
        u, v = para[:2]
        if is_orthogonal(u, v):
            return cls(np.abs(u), np.abs(v), np.angle(u), c)

        # Step 1
        ur = rotate_towards(u, v, 0.25)
        s = midpoint(ur, v)

        # Step 2
        r = rect(np.abs(s), np.angle(ur - s)) + s
        m = rect(np.abs(s), np.angle(v - s)) + s

        a = np.abs(v - r)
        b = np.abs(v - m)

        # graph(np.concatenate([
        #     closed(para + c),
        #     np.array([u, c, v, ur, 0, s, r, c, m]),
        #     Circle.at(np.linspace(0, 1, 500)) * np.abs(s) + s
        #     ]), lines=True)
        return cls(a, b, np.angle(m), c)

    @classmethod
    def fit_points(cls, points, method='halir'):
        """Make an ellipse by fitting a set of points using
        algorithm from Fitzgibbon.
        """
        if method == 'halir':
            fit_method = ellipse_fit_fitzgibbon
        elif method == 'fitzgibbon':
            fit_method = ellipse_fit_halir
        else:
            raise NotImplementedError(f"Method '{method}' not implemented.")
        return cls.from_general_coefficients(*fit_method(points))

    @property
    def general_coefficients(self):
        """The general form coefficients on an ellipse.
        https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        """
        # Helpers
        sin_theta = np.sin(self.angle)
        cos_theta = np.cos(self.angle)
        sin_theta_sq = sin_theta ** 2.0
        cos_theta_sq = cos_theta ** 2.0
        a_sq = self.a ** 2.0
        b_sq = self.b ** 2.0
        x = self.origin.real
        y = self.origin.imag
        # The general coefficient equations
        a = a_sq * sin_theta_sq + b_sq * cos_theta_sq
        b = 2 * (b_sq - a_sq) * sin_theta * cos_theta
        c = a_sq * cos_theta_sq + b_sq * sin_theta_sq
        d = -2.0 * a * x - b * y
        e = -b * x - 2.0 * c * y
        f = a * x ** 2.0 + b * x * y + c * y ** 2.0 - a_sq * b_sq
        return np.array([a, b, c, d, e, f])

    @classmethod
    def from_general_coefficients(cls, a, b, c, d, e, f):
        """
        See formulas at the end of section:
        https://en.wikipedia.org/wiki/Ellipse#Canonical_form
        """
        # TODO Check for degenerate cases described here:
        # https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        den = b ** 2 - 4.0 * a * c
        acb_pythagorean = np.sqrt(((a - c) ** 2) + b ** 2)
        ab_common = a * (e ** 2) + c * (d ** 2) - b * d * e + den * f

        a = -np.sqrt(2.0 * ab_common * (a + c + acb_pythagorean)) / den
        b = -np.sqrt(2.0 * ab_common * (a + c - acb_pythagorean)) / den
        # Make sure a is the major axis
        if float(a) < float(b):
            [a, b] = [b, a]

        x = (2.0 * c * d - b * e) / den
        y = (2.0 * a * e - b * d) / den
        origin = x + 1j * y

        if float(b) == 0:
            theta = 0 if a <= c else pi2 / 4.0
        else:
            theta = np.arctan((c - a - acb_pythagorean) / b)

        return cls(a, b, theta, origin)
