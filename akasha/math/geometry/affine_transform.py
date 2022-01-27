#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Affine Transform module
"""

from __future__ import division

import skimage.transform as skt

from akasha.utils.python import class_name, _super
from akasha.math import as_complex, complex_as_reals


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
        return as_complex(_super(self).__call__(coords).T)

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
        return as_complex(_super(self).inverse(coords).T)

    def __repr__(self):
        return f'{class_name(self)}(' + \
            f'scale={self.scale!r}, ' + \
            f'rotation={self.rotation!r}, ' + \
            f'shear={self.shear!r}, ' + \
            f'translation={self.translation!r})'
