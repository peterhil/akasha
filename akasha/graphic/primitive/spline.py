#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clothoid splines.
"""

from __future__ import division

import numpy as np
import scipy as sc

from cmath import rect

from akasha.graphic.geometry import angle_between, vectors
from akasha.utils.math import as_complex, normalize, rad_to_deg


def clothoid(points):
    """
    The clothoid (or Euler) spiral curve.
    See: http://en.wikipedia.org/wiki/Euler_spiral
    """
    points = np.atleast_1d(points)
    a = np.zeros((2, len(points)))
    a[:, :] = sc.special.fresnel(points)[:]
    return as_complex(a)


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
    return rotated * (scale / 2)


def curvature(previous_pt, point, next_pt):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at:
    http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    (v1, v2) = vectors(previous_pt, point, next_pt)
    return 2 * np.sin(angle_between(v1, v2) / 2) / np.sqrt(np.abs(v1) * np.abs(v2))


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
