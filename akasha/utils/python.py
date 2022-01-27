#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python utilities
"""

def class_name(self):
    return self.__class__.__name__


def _super(self):
    """
    Easier to remember function to get the super class for self (or passed in instance).
    """
    return super(self.__class__, self)
