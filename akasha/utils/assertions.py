#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Assertion utilities
"""

import numpy as np

from akasha.math import map_array


def assert_type(types, *args):
    """
    Assert that all the arguments are instances of the specified types.
    """
    assert np.all(map_array(lambda p: isinstance(p, types), args)), \
        "All arguments must be instances of %s, got:\n%s" % (types, map_array(type, args))
