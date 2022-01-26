#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Timings module.
"""

from akasha.timing.sampler import sampler
from akasha.timing.timed import Timed
from akasha.timing.watch import Watch


def time_slice(dur, start=0, time=False):
    """Use a time slice argument or the provided attributes 'dur' and 'start' to
    construct a time slice object."""
    start *= sampler.rate
    time = time or slice(int(round(0 + start)), int(round(dur * sampler.rate + start)))
    if not isinstance(time, slice):
        raise TypeError("Expected a %s for 'time' argument, got %s." % (slice, type(time)))
    return time
