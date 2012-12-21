#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from akasha.types import assert_type, signed
from akasha.utils.math import complex_as_reals


def line_bresenham(x0, y0, x1, y1, colour=1.0, indices=False):
    """
    Bresenham line drawing algorithm.
    Converted from C version at http://free.pages.at/easyfilter/bresenham.html by Peter H.
    """
    assert_type(signed, x0, y0, x1, y1)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    dx =  np.abs(x1 - x0)
    dy = -np.abs(y1 - y0)
    bx = np.min((x0, x1))
    by = np.min((y0, y1))
    err = dx + dy # error value e_xy
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
        if (e2 >= dy): err += dy; x0 += sx # e_xy + e_x > 0
        if (e2 <= dx): err += dx; y0 += sy # e_xy + e_y < 0

def line_linspace(x0, y0, x1, y1, endpoint=True):
    assert_type(signed, x0, y0, x1, y1)
    size = np.max([np.abs(x1 - x0), np.abs(y1 - y0)]) + int(bool(endpoint))
    return complex_as_reals(np.linspace(x0 + y0 * 1j, x1 + y1 * 1j, size, endpoint=endpoint)).astype(np.int32)

def line_linspace_cx(start, end, resolution=1000, endpoint=True):
    # assert_type(complex, start, end)
    start = complex(start)
    end = complex(end)
    distance = np.abs(start - end)
    size = resolution * np.max([distance.real, distance.imag]) + int(bool(endpoint))
    return np.round(resolution * complex_as_reals(np.linspace(start, end, size, endpoint=endpoint))).astype(np.int32)

