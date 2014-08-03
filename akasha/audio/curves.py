#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curves for oscillators
"""

from __future__ import division

import numpy as np

from cmath import rect

from akasha.audio.generators import PeriodicGenerator
from akasha.graphic.geometry import AffineTransform, is_orthogonal, midpoint, rotate_towards
from akasha.timing import sampler
from akasha.utils import issequence
from akasha.utils.math import as_complex, clip, pi2, normalize
from akasha.utils.patterns import Singleton


class Curve(PeriodicGenerator):
    """Generic curve abstraction"""

    @staticmethod
    def at(points):
        """The curve path at points given."""
        raise NotImplementedError("Please implement static method at() in a subclass.")

    def __call__(self, points):
        return self.at(points)

    def __repr__(self):
        return "%s()" % (self.__class__.__name__,)

    def __str__(self):
        return repr(self)


class Circle(Curve, Singleton):
    """Curve of the circle"""

    @staticmethod
    def at(points):
        """Circle points of the unit circle at the complex plane."""
        return np.exp(1j * pi2 * points)


class Square(Curve, Singleton):
    """Curve of the square wave. Made with np.sign()."""

    @staticmethod
    def at(points):
        return clip(Circle.at(points) * 2)


class Super(Curve):
    """Oscillator curve that has superness parameters."""

    def __init__(self, m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Super oscillator can be initialized using superness parameters to control the shape.

        See 'Superellipse' article at Wikipedia for explanation of what these parameters mean:
        http://en.wikipedia.org/wiki/Superellipse

        Also see Gielis curve:
        http://www.2dcurves.com/power/powergc.html
        http://en.wikipedia.org/wiki/Superformula
        """
        self.superness = self.get_superness(m, n, p, q, a, b)

    @staticmethod
    def get_superness(m=None, n=None, p=None, q=None, a=None, b=None):
        """
        Return a superness out of the arguments m to b.
        If given a sequence as m, will spread to other arguments, ignoring them.

        Defaults to (m=4.0, n=2.0, p=2.0, q=2.0, a=1.0, b=1.0) as an np.array.
        If b is missing, but a provided, it will be given the value of a.
        Arguments n, p and q are handled similarly. Missing values are filled from the left.
        """
        if issequence(m):
            superness = np.repeat(None, 6)
            superness[:min(len(m), 6)] = np.array(m[:6], dtype=np.float64)
            # if len(superness) < 6:
            #     if len(superness) < 4:
            #         superness = pad(superness, count=(4 - len(superness)))
            #     superness = pad(superness, count=(6 - len(superness)), value=1)
            # return superness
            (m, n, p, q, a, b) = superness

        return np.array([
            m or 4.0,
            n or 2.0,
            p or n or 2.0,
            q or p or n or 2.0,
            a or 1.0,
            b or a or 1.0,
        ], dtype=np.float64)

    def at(self, points):
        """Superformula curve at points."""
        return normalize(self.formula(points, self.superness)) * Circle.at(points)

    @staticmethod
    def formula(at, superness):
        """
        Superformula function.
        Generates amplitude curves applicable to oscillators by multiplying.

        Usage:
        supercurve(angles, superness)
        s = Super(431, m, n, p, q, a, b), where
        m is the number of spikes and n-q determine the roundness.

        For more information, see:
        http://en.wikipedia.org/wiki/Superellipse and
        http://en.wikipedia.org/wiki/Superformula
        """
        (m, n, p, q, a, b) = list(superness)
        assert np.isscalar(m), "%s in superformula is not scalar." % m
        coeff = pi2 * at * (m / 4.0)
        return (np.abs(np.cos(coeff) / a) ** p + np.abs(np.sin(coeff) / b) ** q) ** (-1.0 / n)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(self.superness))

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, tuple(self.superness))


def chirp_zeta(z1=-0.5 - 100j, z2=0.5 + 100j, dur=10):
    """
    Chirp sound made by sampling a line z (z1 -> z2) from the complex plane,
    and using the function (k ** -z, k = 0..n) used for summation in the Riemann Zeta function.

    Other interesting values to try:
    chirp_zeta(-10.5-1000j, 1.5-10000j)

    Reference: http://en.wikipedia.org/wiki/Riemann_zeta_function
    """
    n = int(round(dur * sampler.rate))
    z = np.linspace(z1, z2, n)
    k = np.arange(n)
    return normalize(k ** -z)


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

        u, s, v = np.linalg.svd(tr._matrix[:2, :2], full_matrices=False, compute_uv=True)
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
