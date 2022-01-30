#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Geometry module
"""

from __future__ import division

import numpy as np

from akasha.funct.itertools import consecutive
from akasha.math import (
    cartesian,
    distances,
    normalize,
    overlap,
    pad_left,
    pi2,
    repeat,
)


def angle_between(a, b, c=None):
    """Angle between two points (a and c though b) in radians.

    If only a and b is given, will give the angle between
    (a and b through 0).

    Dot Product & Angle Between Vectors:
    http://www.youtube.com/watch?v=p8BZTFNSKIw

    Also see:
    https://en.wikipedia.org/wiki/Vector_dot_product#Geometric_definition
    """
    if c is None:
        b, c = 0, b
    return np.angle(cartesian(1, np.angle(c - b) - np.angle(a - b)))


def angles_between(points, *rest):
    """Angles between each three consecutive points on the complex
    plane in radians.

    If given less than three points, the input is padded from
    the start with the first value.

    For example:
    >>> angles_between([3j, 0])
    array([-1.5707963267948966])
    """
    points = np.append(points, rest).astype(np.complex128)
    if len(points) == 0:
        return points
    points = pad_left(points, points[0], 3)
    return angle_between(*overlap(points, 3))


def triangle_incenter(a, b, c):
    """Calculate the coordinate of the incenter point on a triangle
    given vertices a, b, and c (as complex numbers).

    https://en.wikipedia.org/wiki/Incenter#Cartesian_coordinates
    """
    edges = np.array([b, c, a, b], dtype=np.complex128)
    a_side, b_side, c_side = distances(edges)
    perimeter = a_side + b_side + c_side

    x = (a_side * a.real + b_side * b.real + c_side * c.real) / perimeter
    y = (a_side * a.imag + b_side * b.imag + c_side * c.imag) / perimeter

    return x + y * 1j


def directions(points):
    """Get direction angles (as in compass directions) for
    navigating through a set of points.

    https://en.wikipedia.org/wiki/Direction_(geometry)
    """
    return np.angle(vectors(points))


def turtle_turns(points):
    """Changes in orientation angle on a path formed by points (like
    in turtle graphics).

    This differs from turns, in that the changes are relative to
    previous path segment.

    >>> o = np.array([0.5+0.5j, -0.5+0.5j, -0.5-0.5j,
    0.5-0.5j])

    >>> turtle_turns(o) / pi2

    array([ 0.25,  0.25])
    """
    return np.array(
        [
            np.ediff1d(directions(seg / vectors(seg)[1]))
            for seg in consecutive(points, 3)
        ]
    ).flatten()


def circumcircle_radius(a, b, c):
    """Find the circumcircle of three points.

    Takes into account the counterclockwise (positive radius)
    or clockwise (negative radius) direction, that the points represent.
    """
    if a == b == c:
        return 0
    if is_collinear(a, b, c):
        return np.inf
    side = np.abs(a - c)
    angle = angle_between(a, b, c)
    return side / (2 * -np.sin(angle))


def circumcircle_radius_alt(a, b, c):
    """Find the circumcircle of three points.

    Takes into account the counterclockwise (positive radius) or
    clockwise (negative radius) direction, that the points represent.
    """
    if a == b == c:
        return 0
    if is_collinear(a, b, c):
        return np.inf
    (v1, v2) = np.array([a, c]) - b
    side = np.abs(a - c)
    return side / (2 * -np.sin(angle_between(v1, v2)))


def closed(signal):
    return np.append(signal, signal[0])


def wrap_ends(signal, n=1):
    return np.concatenate([signal[-n:], np.asanyarray(signal), signal[:n]])


def pad_ends(signal, value=0, n=1):
    return np.concatenate(
        (repeat(value, n), np.asanyarray(signal), repeat(value, n))
    )


def repeat_ends(signal, n=1):
    return np.concatenate(
        [repeat(signal[:1], n), np.asanyarray(signal), repeat(signal[-1:], n)]
    )


def is_collinear(a, b, c):
    """Return true if the three points are collinear.

    For other methods, see:
    https://en.wikipedia.org/wiki/Collinearity#Collinearity_of_points_whose_coordinates_are_given
    """
    if a == b or b == c or a == c:
        return True
    return angle_between(a, b, c) % np.pi == 0


def is_orthogonal(a, b, c=0):
    """Return true if two complex points (a, b) are orthogonal from
    center point (c).
    """
    return np.abs(angle_between(a, c, b)) == np.pi / 2


def midpoint(a, b):
    """Midpoint is the middle point of a line segment."""
    return ((a - b) / 2.0) + b


def orient(arr, end=1 + 0j, inplace=False):
    """Orientates (or normalizes) an array by translating startpoint
    to the origo, and scales and rotates endpoint to the parameter 'end'.
    """
    if inplace:
        arr -= arr[0]
        arr *= end / arr[-1]
        return arr
    else:
        return (arr - arr[0]) * (end / (arr[-1] - arr[0]))


def parallelogram_point(a, b, c):
    """Make a parallelogram out of a triangle.

    In other words, rotate point b pi degrees around midpoint of
    line form a to c.
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


def rotate(z, tau):
    return z * np.exp(tau * pi2 * 1j)


def rotate_towards(u, v, tau, center=0):
    """
    Rotate point u tau degrees *towards* v around center.
    """
    s, t = np.array([u, v]) - center
    sign = -1 if (np.angle(s) - np.angle(t)) % pi2 > np.pi else 1
    return s * (-np.exp(pi2 * 1j * tau) * sign) + center


def vectors_from_origo(points, origo=0):
    return np.asarray(points) - origo


def vectors(points, *rest):
    """Get the vectors that give directions on how to move through
    some points.

    You could use this to move something in way the turtle
    graphics in the Logo programming language works.

    https://en.wikipedia.org/wiki/Logo_(programming_language)
    https://en.wikipedia.org/wiki/Turtle_graphics
    """
    points = np.append(points, rest)
    return np.ediff1d(points)
