#!/usr/bin/env python
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
    assert np.all(map_array(lambda p: isinstance(p, types), args)), (
        f'All arguments must be instances of {types}, '
        + f'got:\n{map_array(type, args)}'
    )
