#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry module
"""

from __future__ import division

import numpy as np
import skimage.transform as skt

from akasha.utils import _super
from akasha.utils.math import as_complex, complex_as_reals, pi2


class AffineTransform(skt.AffineTransform):
    """
    Affine transformation on the complex plane.

    Inherits from skimage.transform.AffineTransform:
    http://scikit-image.org/docs/0.8.0/api/skimage.transform.html#affinetransform

    Adds complex_plane method to transform comples signals.

    References:
    Postscript Language Reference Manual, 3rd edition, chapters "4.3.2 Transformations" and
    "4.3.3 Matrix Representation and Manipulation"
    http://www.adobe.com/products/postscript/pdfs/PLRM.pdf
    """
    def __call__(self, signal):
        """
        Apply the affine transformation onto a signal on the complex plane.
        """
        coords = complex_as_reals(signal).T
        return as_complex(self._apply_mat(coords, self._matrix).T)

    def estimate(self, src, dst):
        """
        Estimate the required affine transformation on a complex plane from src to dst.
        """
        src = complex_as_reals(src).T
        dst = complex_as_reals(dst).T
        _super(self).estimate(src, dst)

    def inverse(self, signal):
        """
        Apply the inverse affine transformation onto a signal on the complex plane.
        """
        coords = complex_as_reals(signal).T
        return as_complex(self._apply_mat(coords, self._inv_matrix).T)

    def __repr__(self):
        return "{}(scale={}, rotation={}, shear={}, translation={})".format(
            self.__class__.__name__,
            tuple(self.scale),
            float(self.rotation),
            float(self.shear),
            tuple(self.translation),
        )


def is_orthogonal(a, b, c=0):
    """
    Return true if two complex points (a, b) are orthogonal from center point (c).
    """
    return np.abs(np.angle(a - c) - np.angle(b - c)) == np.pi / 2


def rotate_towards(u, v, tau, center=0):
    """
    Rotate point u tau degrees *towards* v around center.
    """
    s, t = np.array([u, v]) - center
    sign = -1 if (np.angle(s) - np.angle(t)) % pi2 > np.pi else 1
    return s * (-np.exp(pi2 * 1j * tau) * sign) + center
