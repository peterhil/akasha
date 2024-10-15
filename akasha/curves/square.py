"""
Square curve
"""

from akasha.curves.circle import Circle
from akasha.curves.curve import Curve
from akasha.math import clip
from akasha.utils.patterns import Singleton


class Square(Curve, Singleton):
    """Curve of the square wave. Made with np.sign()."""

    @staticmethod
    def at(points):
        return clip(Circle.at(points) * 2)
