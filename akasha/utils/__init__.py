#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for Akasha
"""


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


def trace_c(frame, event, arg):
    """
    Trace C calls for debugging.

    Usage: sys.settrace(trace_c)
    """
    if event == 'c_call' or arg is not None and 'IPython' not in frame.f_code.co_filename:
        print("%s, %s: %d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
