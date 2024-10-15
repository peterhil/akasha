"""
Curvature module
"""

import numpy as np

from akasha.curves import Ellipse
from akasha.funct.itertools import consecutive
from akasha.math.geometry import (
    # circumcircle_radius,
    circumcircle_radius_alt,
    is_collinear,
)
from akasha.math import all_equal, complex_as_reals, div_safe_zero, pi2


def circle_curvature(a, b, c):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at:
    http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    return div_safe_zero(1, circumcircle_radius_alt(a, b, c))


def estimate_curvature(signal):
    return np.array(
        [circle_curvature(*points) for points in consecutive(signal, 3)]
    )


def ellipse_curvature(para, debug=False):
    """
    Fit ellipse from three points using conjugate diameters.
    """
    # TODO: Investigate fitting ellipses from five points.
    # https://math.stackexchange.com/questions/151969/ellipse-fitting-methods
    points = para[:3]
    if all_equal(points):
        return np.inf
    if is_collinear(*points):
        return 0
    ell = Ellipse.from_conjugate_diameters(points)

    if debug:
        import pylab as lab
        from akasha.graphic.plotting import plotting

        with plotting():
            lab.plot(*complex_as_reals(ell.at(np.linspace(0, 1, 200))))

    return ell.curvature(np.angle(points[0] - ell.origin) / pi2)


def estimate_curvature_with_ellipses(signal, debug=False):
    res = np.array(
        [
            ellipse_curvature(points, debug)
            for points in consecutive(signal, 3)
        ]
    )

    if debug:
        import pylab as lab
        from akasha.graphic.plotting import plotting

        with plotting():
            lab.interactive(True)
            lab.plot(*complex_as_reals(signal), color='k')
            lab.axis('equal')

    return res
