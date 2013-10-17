#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clothoid splines.
http://www.dgp.toronto.edu/~karan/papers/sbim2008mccrae.pdf
"""

from __future__ import division

import numpy as np
import scipy as sc

from cmath import rect

from akasha.audio.curves import Ellipse
from akasha.funct import consecutive
from akasha.graphic.geometry import circumcircle_radius, is_collinear
from akasha.utils.math import as_complex


def clothoid(points, scaled=True):
    """
    The clothoid (or Euler) spiral curve.
    Calculated using the Fresnel integrals.

    See: http://en.wikipedia.org/wiki/Euler_spiral
    """
    points = np.atleast_1d(points)
    if scaled:
        k = np.sqrt(2.0 / np.pi)
        s, c = sc.special.fresnel(k * points) / k
    else:
        s, c = sc.special.fresnel(points)
    return as_complex(np.array([c, s]))


def clothoid_slice(start, stop, n, endpoint=False):
    """
    A slice of clothoid spiral.
    """
    return clothoid(np.linspace(start, stop, n, endpoint))


def clothoid_windings(start, stop, n, endpoint=False):
    """
    A piece of clothoid with start and stop being winding values.
    """
    return clothoid(np.linspace(wind(start), wind(stop), n, endpoint))


def clothoid_length(turn, diff=0.5, n=1000, endpoint=True):
    """
    A piece of clothoid with starting point at turn windings (positive or negative)
    and going there to diff windings point.
    """
    return clothoid_windings(turn, turn + diff, n, endpoint)


def wind(x):
    """
    Normalizes points on clothoid to have x number of turns (windings) wrt. to origo
    """
    return np.sign(x) * np.sqrt(np.abs(x) * 4)


def clothoid_angle(s):
    """
    Find the tangent angle of the clothoid curve at s points.
    """
    s = np.atleast_1d(s)
    a = 1.0 / np.sqrt(2.0)
    return np.sign(s) * ((a * s) ** 2.0) * np.pi


def cl_piece(start, stop, n, endpoint=False, scale=1, norm=False, fn=clothoid_windings):
    """
    Takes a piece of clothoid and rotates and translates it to unit vector.
    """
    angle = clothoid_angle(wind(start))
    curve = fn(start, stop, n, endpoint)
    coeff = 1 / np.abs(curve[-1]) if norm else 1
    rotated = (curve - curve[0]) * rect(coeff, angle)
    return rotated * scale


def curvature(a, b, c):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at:
    http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    if a == b == c:
        return np.inf
    if is_collinear(a, b, c):
        return 0
    return 1 / circumcircle_radius(a, b, c)


def estimate_curvature_circle(signal):
    return np.array([curvature(*points) for points in consecutive(signal, 3)])


def ellipse_curvature(pts):
    ell = Ellipse.from_conjugate_diameters(pts[:3])
    return ell.curvature(np.angle(pts[1] - midpoint(pts[0], pts[2])) / pi2)


def estimate_curvature(signal):
    return np.array([ellipse_curvature(points) for points in consecutive(signal, 3)])


# Circular arcs

def arc(points, s=1.0):
    """
    An arc is an arc.

    Parameters
    ==========
    points:
        An array of points at which to sample the arc.
    s:
        The curvature parameter.
    """
    points = np.atleast_1d(points)
    return (1.0 / s) * (np.exp(1j * np.pi * 2.0 * points) - 1.0)


# Line segments

def line(a, b, n=100):
    """
    Basic line between two points.
    """
    return np.linspace(a, b, n)


def linediff(a, d, n=100):
    """
    Line with a point and a vector.
    """
    return line(a, a + d, n)
