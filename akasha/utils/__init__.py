#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for Akasha
"""

import os


system = os.uname().sysname
open_cmd = 'open' if system == 'Darwin' else 'xdg-open'


def _super(self):
    """
    Easier to remember function to get the super class for self (or passed in instance).
    """
    return super(self.__class__, self)
