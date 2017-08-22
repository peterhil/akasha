#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Ellipse fitting module
"""

from __future__ import division

import numpy as np
import numpy.linalg as la

from akasha.math import complex_as_reals


def ellipse_fit_fitzgibbon(points):
    """
    Direct Least Squares Fitting of Ellipses
    by Andrew W. Fitzgibbon et al.

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.897&rep=rep1&type=pdf
    http://cseweb.ucsd.edu/~mdailey/Face-Coord/ellipse-specific-fitting.pdf
    https://se.mathworks.com/matlabcentral/fileexchange/22684-ellipse-fit--direct-method-
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ellipse-pami.pdf

    Returns the general form ellipse coefficients (A, B, C, D, E, F):
    https://en.wikipedia.org/wiki/Ellipse#General_ellipse
    """
    # TODO: Also check improvements by Radim Halir & Jan Flusser:
    # http://autotrace.sourceforge.net/WSCG98.pdf

    x, y = complex_as_reals(points)
    size = len(x)

    # Build design matrix
    design = np.matrix([
        x * x,
        x * y,
        y * y,
        x,
        y,
        np.ones(size),
    ]).T

    # Build scatter matrix
    scatter = np.dot(design.T, design)

    # Build 6x6 constraint matrix
    constraint = np.zeros([6, 6])
    constraint[0, 2] = constraint[2, 0] = 2
    constraint[1, 1] = -1

    # Solve eigensystem
    [gevalues, gevector] = la.eig(la.inv(scatter) * constraint)
    gevalues = np.diag(gevalues)

    # Find positive eigenvalue
    [pos_row, pos_column] = np.where((gevalues > 0) & (np.bitwise_not(np.isinf(gevalues))))

    # Extract eigenvector corresponding to positive eigenvalue.
    # These are the general form of coefficients for ellipse (A..F).
    return np.squeeze(np.asarray(gevector[:, pos_column], dtype=np.float64))
