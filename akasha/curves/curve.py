#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base periodic curve module
"""

from __future__ import division

from akasha.audio.generators import PeriodicGenerator


class Curve(PeriodicGenerator):
    """Generic curve abstraction"""

    @staticmethod
    def at(points):
        """The curve path at points given."""
        raise NotImplementedError("Please implement static method at() in a subclass.")

    def __call__(self, points):
        return self.at(points)

    def __repr__(self):
        return "%s()" % (self.__class__.__name__,)

    def __str__(self):
        return repr(self)


