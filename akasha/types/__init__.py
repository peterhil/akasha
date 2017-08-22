#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Types module
"""

import numpy as np

from akasha.math import map_array


colour_values = np.float32
colour_result = np.uint8

signed = (int, float, np.signedinteger, np.floating)


def assert_type(types, *args):
    """
    Assert that all the arguments are instances of the specified types.
    """
    assert np.all(map_array(lambda p: isinstance(p, types), args)), \
        "All arguments must be instances of %s, got:\n%s" % (types, map_array(type, args))
