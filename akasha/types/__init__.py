#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

colour_values = np.float32
colour_result = np.float32

signed = (int, float, np.signedinteger, np.floating)

def assert_type(types, *args):
    assert np.all(map(lambda p: isinstance(p, types), args)), \
        "All arguments must be instances of %s, got:\n%s" % (types, map(type, args))

