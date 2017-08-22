#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graphic line drawing functions.
"""

import numpy as np

from akasha.types import assert_type, signed
from akasha.math import complex_as_reals

from skimage import draw as skdraw


def bresenham(coords):
    """
    Return coordinates for a line using the bresenham algorithm.

    Wraps the function from scikits-image (skimage), so that it's usable with map and numpy.
    The original function is written using pyrex, so it's very fast.
    """
    return np.array(skdraw.bresenham(*coords), dtype=np.uint32)

def line_bresenham(x0, y0, x1, y1, colour=1.0, indices=False):
    """
    Bresenham line drawing algorithm.
    Converted from C version at http://free.pages.at/easyfilter/bresenham.html by Peter H.
    """
    assert_type(signed, x0, y0, x1, y1)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    dx = np.abs(x1 - x0)
    dy = -np.abs(y1 - y0)
    bx = np.min((x0, x1))
    by = np.min((y0, y1))
    err = dx + dy  # error value e_xy
    if indices:
        out = []
    else:
        out = np.zeros((-dy + 1, dx + 1))

    while True:
        if indices:
            out.append((x0, y0))
        else:
            out[y0 - by, x0 - bx] = colour

        if (x0 == x1 and y0 == y1):
            return np.array(out).T

        e2 = 2 * err

        if (e2 >= dy):
            err += dy  # e_xy + e_x > 0
            x0 += sx

        if (e2 <= dx):
            err += dx  # e_xy + e_y < 0
            y0 += sy


def line_linspace(x0, y0, x1, y1, endpoint=True):
    """
    Draw a line using np.linspace using real x and y coordinates.
    """
    assert_type(signed, x0, y0, x1, y1)
    size = np.max([np.abs(x1 - x0), np.abs(y1 - y0)]) + int(bool(endpoint))
    points = complex_as_reals(np.linspace(x0 + y0 * 1j, x1 + y1 * 1j, size, endpoint=endpoint))
    return points.astype(np.int32)


def line_linspace_cx(start, end, resolution=1000, endpoint=True):
    """
    Draw a line using np.linspace from a start point to an end point (both on the complex plane).
    """
    # assert_type(complex, start, end)
    start = complex(start)
    end = complex(end)
    distance = np.abs(start - end)
    size = resolution * np.max([distance.real, distance.imag]) + int(bool(endpoint))
    points = complex_as_reals(np.linspace(start, end, size, endpoint=endpoint))
    return np.round(resolution * points).astype(np.int32)
