#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

from types import FunctionType, CodeType


def funcprinter(func):
    for feat in dir(func.func_code):
        if feat[0:3] == 'co_':
            print feat, '=>', func.func_code.__getattribute__(feat)

def macro(func):
    """Lisp style macros for Python"""
    # if inspect.getdourcefile(func): # OR
    # try:
    #     code_str = inspect.getsource(func)   # Func must be defined in a file
    # expect IOError(e):
    #     return None
    pass

