#!/usr/bin/env python

"""
Array utilities
"""

import numpy as np


def is_sequence(arg):
    """
    Checks if arg is a sequence.
    """
    # For discussion, see:
    # http://stackoverflow.com/questions/1835018/python-check-if-an-object-is-a-list-or-tuple-but-not-string/1835259#1835259
    return (
        not hasattr(arg, "strip")
        and hasattr(arg, "__getitem__")
        or hasattr(arg, "__iter__")
    )


def is_empty(signal):
    """
    Return true if signal is empty.
    """
    return np.asanyarray(signal).size == 0


def is_silence(signal):
    """
    Return true if signal is empty or all zeros.
    """
    return np.all(np.asanyarray(signal) == 0)


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        pass  # shape was not a number
    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass  # shape was not iterable
    raise TypeError('shape must be an int, or a tuple of ints')
