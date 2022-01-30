#!/usr/bin/env python
# encoding: utf-8

"""
Releasable mixin module
"""

from akasha.audio.mixins.releasable import Releasable
from akasha.audio.mixins.tuneable import Tuneable


class Playable(Tuneable, Releasable):
    """
    Mixin to create playable composite sound object.
    """

    pass
