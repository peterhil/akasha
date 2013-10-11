#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry module
"""

from __future__ import division

import numpy as np
import skimage.transform as skt

from akasha.utils import _super
from akasha.utils.math import as_complex, complex_as_reals


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
