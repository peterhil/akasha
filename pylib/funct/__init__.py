#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from itertools import tee, izip
import numpy as np

# itertools recipes -- http://docs.python.org/library/itertools.html#recipes
def take(n, iterable):
    "Return first n items of the iterable as a Numpy Array"
    return np.fromiter(islice(iterable, n), dtype=iterable.dtype)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)
