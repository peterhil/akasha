#!/usr/bin/env python

"""
Debug utilities
"""


def trace_c(frame, event, arg):
    """
    Trace C calls for debugging.

    Usage: sys.settrace(trace_c)
    """
    if (
        event == 'c_call'
        or arg is not None
        and 'IPython' not in frame.f_code.co_filename
    ):
        print(f'{event}, {frame.f_code.co_filename}: {frame.f_lineno}')
    return trace
