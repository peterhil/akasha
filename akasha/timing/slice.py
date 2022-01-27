#!/usr/bin/env python
# -*- coding: utf-8 -*-


def time_slice(dur, start=0, time=False):
    """Use a time slice argument or the provided attributes 'dur'
    and 'start' to construct a time slice object.
    """
    start *= sampler.rate
    time = time or slice(
        int(round(0 + start)),
        int(round(dur * sampler.rate + start))
    )
    if not isinstance(time, slice):
        raise TypeError(
            f"Expected a {slice} for 'time' argument, got {type(time)}."
        )
    return time
