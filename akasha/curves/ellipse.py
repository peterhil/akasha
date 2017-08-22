#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ellipses module
"""

from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy as sc

from cmath import rect

from akasha.curves.curve import Curve
from akasha.curves.ellipse_fit import ellipse_fit_fitzgibbon
from akasha.math import complex_as_reals
from akasha.math.geometry import is_orthogonal, midpoint, rotate_towards
from akasha.math.geometry.affine_transform import AffineTransform
from akasha.math import pi2


class Ellipse(Curve):
    """
    Ellipse curve
    """
    def __init__(self, a, b, angle=0, origin=0):
        self.a = a
        self.b = b
        self.angle = angle
        self.origin = origin

    def __repr__(self):
        return "%s(%r, %r, %r, %r)" % (self.__class__.__name__, self.a, self.b, self.angle, self.origin)

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
        x = self.origin.real + cos * np.cos(self.angle) - sin * np.sin(self.angle)
        y = self.origin.imag + cos * np.sin(self.angle) + sin * np.cos(self.angle)
        return as_complex(np.array([np.asanyarray(x), np.asanyarray(y)]))

    def at(self, tau):
        """
        Polar form of ellipse relative to center, translated and rotated to origin and angle.
        https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
        """
        thetas = np.asanyarray(tau) * pi2
        radius = self.a * self.b / np.sqrt((self.b * np.cos(thetas)) ** 2 + (self.a * np.sin(thetas)) ** 2)
        return radius * np.exp((thetas + self.angle) * 1j) + self.origin

    def curvature(self, tau):
        """
        Curvature of an ellipse.
        http://mathworld.wolfram.com/Ellipse.html formula 59
        """
        t = np.asanyarray(tau) * pi2 + self.angle
        return (self.a * self.b) / (self.b ** 2 * np.cos(t) ** 2 + self.a ** 2 * np.sin(t) ** 2) ** (3 / 2)

    def roc(self, tau):
        """
        Radius of curvature.
        """
        return 1.0 / self.curvature(tau)

    def arc_length(self, tau):
        """
        Arc length of the ellipse.
        Formula (4) from: http://paulbourke.net/geometry/ellipsecirc/Abbott.pdf
        """
        rad = np.fmod(np.asarray(tau) * pi2 - self.angle, np.pi)  # TODO is substracting self.angle necessary?
        return self.a * sc.special.ellipeinc(rad, self.eccentricity ** 2)

    @property
    def circumference(self):
        return 4.0 * self.arc_length(0.25)

    @property
    def eccentricity(self):
        """
        Eccentricity of the ellipse: https://en.wikipedia.org/wiki/Ellipse#Eccentricity
        """
        a, b = self.a, self.b
        if a < b: a, b = b, a
        return np.sqrt(1 - (b / a) ** 2)

    @classmethod
    def from_rhombus(cls, para):
        a, b, c, d = para
        para_origin =  para - midpoint(a, c)
        k, l = np.abs(para_origin)[:2]
        return cls(l, k, np.angle(para_origin)[3], midpoint(a, c))

    @classmethod
    def from_parallelogram(cls, para):
        dia = np.array([1, 1j, -1, -1j])
        sq = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        center = midpoint(para[0], para[2])
        para_at_origin = para - center

        tr = AffineTransform()
        tr.estimate(dia, para_at_origin)

        u, s, v = np.linalg.svd(tr.params[:2, :2], full_matrices=False, compute_uv=True)
        a, b = s[:2]

        uv = np.eye(3); uv[:2, :2] = u * np.diag(s) * v
        uv = AffineTransform(uv)

        return cls(a, b,
                   # np.angle(uv(dia[0])),
                   # np.angle(tr.inverse(dia))[0],
                   # pi2 / 4 + tr.rotation + np.tan(tr.shear),
                   np.angle(tr(sq)[1]),
                   center)

    @classmethod
    def from_conjugate_diameters(cls, para):
        """
        Find the major and minor axes of an ellipse from a parallelogram determining the conjugate diameters.

        Uses Rytz's construction for algorithm:
        http://de.wikipedia.org/wiki/Rytzsche_Achsenkonstruktion#Konstruktion
        """
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
        l = rect(np.abs(s), np.angle(v - s)) + s

        a = np.abs(v - r)
        b = np.abs(v - l)

        # graph(np.concatenate([
        #     closed(para + c),
        #     np.array([u, c, v, ur, 0, s, r, c, l]),
        #     Circle.at(np.linspace(0, 1, 500)) * np.abs(s) + s
        #     ]), lines=True)
        return cls(a, b, np.angle(l), c)

    @classmethod
    def from_points(cls, points):
        """
        Make an ellipse by fitting a set of points.
        """
        return cls.from_general_coefficients(*ellipse_fit_fitzgibbon(points))

    @classmethod
    def from_general_coefficients(cls, a, b, c, d, e, f):
        """
        See formulas at the end of section:
        https://en.wikipedia.org/wiki/Ellipse#Canonical_form
        """
        # TODO Check for degenerate cases described here:
        # https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        den = (b ** 2 - 4.0 * a * c)
        acb_pythagorean = np.sqrt(((a - c) ** 2) + b ** 2)
        ab_common = (a * (e ** 2) + c * (d ** 2) - b * d * e + den * f)

        a = -np.sqrt(2.0 * ab_common * (a + c + acb_pythagorean)) / den
        b = -np.sqrt(2.0 * ab_common * (a + c - acb_pythagorean)) / den
        # Make sure a is the major axis
        if a < b:
            [a, b] = [b, a]

        x = (2.0 * c * d - b * e) / den
        y = (2.0 * a * e - b * d) / den
        origin = x + 1j * y

        if b == 0:
            theta = 0 if a <= c else pi2 / 4.0
        else:
            theta = np.arctan((c - a - acb_pythagorean) / b)

        return cls(a, b, theta, origin)
