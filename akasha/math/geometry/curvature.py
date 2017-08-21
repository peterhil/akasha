#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curvature module
"""

from __future__ import division

import numpy as np

from akasha.curves import Ellipse
from akasha.funct import consecutive
from akasha.math.geometry import circumcircle_radius, circumcircle_radius_alt, is_collinear, pad_ends, repeat_ends, wrap_ends
from akasha.utils.math import all_equal, div_safe_zero, pi2


def circle_curvature(a, b, c):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at:
    http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    return div_safe_zero(1, circumcircle_radius_alt(a, b, c))


def estimate_curvature(signal):
    return np.array([circle_curvature(*points) for points in consecutive(signal, 3)])


def ellipse_curvature(para):
    if all_equal(para[:3]):
        return np.inf
    if is_collinear(*para):
        return 0
    ell = Ellipse.from_conjugate_diameters(para[:3])
    return ell.curvature(np.angle(para[1] - ell.origin) / pi2)


def estimate_curvature_with_ellipses(signal, ends='open'):
    if ends == 'open':
        pass
    elif ends == 'pad':
        signal = pad_ends(signal, 0)
    elif ends == 'repeat':
        signal = repeat_ends(signal)
    elif ends == 'closed':
        signal = wrap_ends(signal)
    else:
        raise NotImplementedError('Unknown method for handling ends')
    return np.array([ellipse_curvature(points) for points in consecutive(signal, 3)])
