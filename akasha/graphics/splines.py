#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import scipy as sc

from cmath import rect


def as_complex(a):
    return a.transpose().flatten().view(np.complex128)

def clothoid(points):
    points = np.atleast_1d(points)
    a = np.zeros((2, len(points)))
    a[:,:] = sc.special.fresnel(points)[:]
    return as_complex(a)

def clothoid_slice(start, stop, n, endpoint=False):
    return clothoid(np.linspace(start, stop, n, endpoint))

def clothoid_windings(start, stop, n, endpoint=False):
    return clothoid(np.linspace(wind(start), wind(stop), n, endpoint))

def clothoid_length(turn, diff=0.5, n=1000, endpoint=True):
    return clothoid_windings(turn, turn+diff, n, endpoint)

def wind(x):
    "Normalizes points on clothoid to have x number of turns (windings) wrt. to origo"
    return np.sign(x) * np.sqrt(np.abs(x)*4)

def clothoid_angle(s):
    s = np.atleast_1d(s)
    a = 1.0/np.sqrt(2.0)
    return np.sign(s) * ((a*s)**2.0) * np.pi

def cl_piece(start, stop, n, endpoint=False, scale=1, norm=False, fn=clothoid_windings):
    """Takes a piece of clothoid and rotates and translates it to unit vector."""
    angle = clothoid_angle(wind(start))
    curve = fn(start, stop, n)
    coeff = 1/np.abs(curve[-1]) if norm else 1
    rotated = (curve - curve[0]) * rect(coeff, angle)
    return rotated * (scale / 2)

def orient(arr, end=1+0j, inplace=False):
    """Orientates (or normalizes) an array by translating startpoint to the origo, and scaling and rotating endpoint to the parameter 'end'."""
    if inplace:
        arr -= arr[0]
        arr *= end/arr[-1]
        return arr
    
    else:
        return ((arr - arr[0]) * (end/(arr[-1] - arr[0])))

def angle_between(a, b):
    """
    Angle in radians between two non-zero vectors.

    http://en.wikipedia.org/wiki/Vector_dot_product#Geometric_interpretation
    """
    return np.angle(np.arccos((a / np.abs(a)) * (b / np.abs(b))))

def angle_between2(a, b):
    # vdot is for complex numbers, but this seems to not work correctly?!
    return np.real(np.vdot(a, b)) / (np.abs(a) * np.abs(b))

def curvature(prev, p, next):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    v1 = p - prev
    v2 = next - p
    print polar(v1), polar(v2), deg(angle_between(v1, v2))
    return 2 * np.sin(angle_between2(v1, v2) / 2)  /  np.sqrt(np.abs(v1) * np.abs(v2))

def circumcircle_radius(a, b, c):
    a0 = a - c
    b0 = b - c
    return np.abs(a0 - b0) / 2 * np.sin(angle_between(a0, b0))

# Circular arcs

def arc(points, curvature=1.0):
    points = np.atleast_1d(points)
    return (1.0/curvature) * (np.exp(1j * np.pi * 2.0 * points) - 1.0)
    
# Line segments

def line(a, b, n=100):
    return np.linspace(a, b, n)

def linediff(a, d, n=100):
    return line(a, a+d, n)