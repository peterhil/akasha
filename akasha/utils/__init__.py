#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for Akasha
"""

import os
import numpy as np


system = os.uname().sysname
open_cmd = 'open' if system == 'Darwin' else 'xdg-open'


def _super(self):
    """
    Easier to remember function to get the super class for self (or passed in instance).
    """
    return super(self.__class__, self)


def issequence(arg):
    """
    Checks if arg is a sequence.

    For discussion, see:
    http://stackoverflow.com/questions/1835018/
    python-check-if-an-object-is-a-list-or-tuple-but-not-string/1835259#1835259
    """
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


def trace_c(frame, event, arg):
    """
    Trace C calls for debugging.

    Usage: sys.settrace(trace_c)
    """
    if event == 'c_call' or arg is not None and 'IPython' not in frame.f_code.co_filename:
        print("%s, %s: %d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
