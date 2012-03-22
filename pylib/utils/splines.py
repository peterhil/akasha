#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import scipy as sc


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

def orient(arr, end=1+0j, inplace=False):
    """Orientates (or normalizes) an array by translating startpoint to the origo, and scaling and rotating endpoint to the parameter 'end'."""
    if inplace:
        arr -= arr[0]
        arr *= end/arr[-1]
        return arr
    
    else:
        return ((arr - arr[0]) * (end/(arr[-1] - arr[0])))
    

# Circular arcs

def arc(points, curvature=1.0):
    points = np.atleast_1d(points)
    return (1.0/curvature) * (np.exp(1j * np.pi * 2.0 * points) - 1.0)
    
# Line segments

def line(a, b, n=100):
    return np.linspace(a, b, n)

def linediff(a, d, n=100):
    return line(a, a+d, n)