#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry module
"""

from __future__ import division

import numpy as np
import skimage.transform as skt

from akasha.utils import _super
from akasha.utils.math import as_complex, complex_as_reals, pi2


class AffineTransform(skt.AffineTransform):
    """
    Affine transformation on the complex plane.

    Inherits from skimage.transform.AffineTransform:
    http://scikit-image.org/docs/0.8.0/api/skimage.transform.html#affinetransform

    Adds complex_plane method to transform comples signals.

    References:
    Postscript Language Reference Manual, 3rd edition, chapters "4.3.2 Transformations" and
    "4.3.3 Matrix Representation and Manipulation"
    http://www.adobe.com/products/postscript/pdfs/PLRM.pdf
    """
    def __call__(self, signal):
        """
        Apply the affine transformation onto a signal on the complex plane.
        """
        coords = complex_as_reals(signal).T
        return as_complex(self._apply_mat(coords, self._matrix).T)

    def estimate(self, src, dst):
        """
        Estimate the required affine transformation on a complex plane from src to dst.
        """
        src = complex_as_reals(src).T
        dst = complex_as_reals(dst).T
        _super(self).estimate(src, dst)

    def inverse(self, signal):
        """
        Apply the inverse affine transformation onto a signal on the complex plane.
        """
        coords = complex_as_reals(signal).T
        return as_complex(self._apply_mat(coords, self._inv_matrix).T)

    def __repr__(self):
        return "{}(scale={}, rotation={}, shear={}, translation={})".format(
            self.__class__.__name__,
            tuple(self.scale),
            float(self.rotation),
            float(self.shear),
            tuple(self.translation),
        )


def angle_between(a, b):
    """
    Angle in radians between two non-zero vectors.
    See: http://en.wikipedia.org/wiki/Vector_dot_product#Geometric_interpretation
    """
    return np.angle(a) - np.angle(b)


def angle_between_dotp(a, b):
    """
    Angle in radians between two nonzero vectors.
    Returns only positive values.

    See:
    http://en.wikipedia.org/wiki/Vector_dot_product#Geometric_interpretation
    http://en.wikipedia.org/wiki/Inner_product
    http://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    """
    # vdot is for complex numbers
    dotp = np.real(np.vdot(a, b))
    return np.arccos(dotp / (np.abs(a) * np.abs(b)))


def circumcircle_radius(a, b, c):
    """
    Find the circumcircle of three points.
    """
    side = np.abs(a - c)
    angle = angle_between(a - b, c - b)
    return np.abs(side / (2 * np.sin(angle)))


def circumcircle_radius_alt(previous_pt, point, next_pt):
    """
    Find the circumcircle of three points.
    """
    (v1, v2) = vectors(previous_pt, point, next_pt)
    return np.abs(v1 - v2) / 2 * np.sin(angle_between(v1, v2))


def closed(signal):
    return np.append(signal, signal[0])


def is_orthogonal(a, b, c=0):
    """
    Return true if two complex points (a, b) are orthogonal from center point (c).
    """
    return np.abs(np.angle(a - c) - np.angle(b - c)) == np.pi / 2


def midpoint(a, b):
    """
    Midpoint is the middle point of a line segment.
    """
    return ((a - b) / 2.0) + b


def orient(arr, end=1 + 0j, inplace=False):
    """
    Orientates (or normalizes) an array by translating startpoint to the origo,
    and scales and rotates endpoint to the parameter 'end'.
    """
    if inplace:
        arr -= arr[0]
        arr *= end / arr[-1]
        return arr
    else:
        return ((arr - arr[0]) * (end / (arr[-1] - arr[0])))


def parallelogram_point(a, b, c):
    """
    Make a parallelogram out of a triangle.
    In other words, rotate point b pi degrees around midpoint of line form a to c.
    """
    m = midpoint(a, c)
    return m - (b - m)


def random_points(n=1, x=1, y=1):
    a, b = np.random.rand(2, n)
    return a * x + b * y * 1j


def random_triangle(x=1, y=1):
    return random_points(3, x, y)


def random_parallelogram(x=1, y=1):
    tri = random_triangle(x, y)
    return normalize(np.append(tri, parallelogram_point(*tri)))


def rotate_towards(u, v, tau, center=0):
    """
    Rotate point u tau degrees *towards* v around center.
    """
    s, t = np.array([u, v]) - center
    sign = -1 if (np.angle(s) - np.angle(t)) % pi2 > np.pi else 1
    return s * (-np.exp(pi2 * 1j * tau) * sign) + center


def vectors(previous_pt, point, next_pt):
    return np.array([previous_pt, next_pt]) - point
