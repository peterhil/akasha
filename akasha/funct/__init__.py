#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functional programming utility functions.
"""

import collections
import numpy as np
import re

from itertools import count, islice, izip, tee

from akasha.utils.log import logger


# function objects - or a kind of function composition?

def _findvars(*function_sources):
    """
    Find variables from a function source code.
    """
    idents = re.compile(r'([a-zA-Z_]\w*)')  # (letter|"_") (letter | digit | "_")*
    nfuncs = {}
    for func in function_sources:
        variables = tuple(set(idents.findall(func)))
        if len(variables) == 1:
            variables = variables[0]
        nfuncs.update([[variables, func]])
    return nfuncs


class fn(object):
    """
    Functor, a function object.
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __call__(self, *args, **kwargs):
        for func in self.__dict__.values():
            func(*args, **kwargs)

    def __repr__(self):
        return repr(self.__dict__)


# itertools recipes -- http://docs.python.org/library/itertools.html#recipes

def take(n, iterable):
    """
    Take first n items of the iterable. Return them as a Numpy array.
    """
    return np.fromiter(islice(iterable, n), dtype=iterable.dtype)


def pairwise(iterable):
    """
    Pair consecutive elements.

    pairwise([s0, s1, s2, s3, ...]) -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def consume(iterator, n=None):
    """
    Advance the iterator n-steps ahead. If n is none, consume entirely.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


# Other iterators and generators

def count_step(start=0, step=1):
    """
    Get an iterator that advances from start with step at a time.
    """
    g = (start + step * i for i in count())
    yield g.next()
    while True:
        yield g.next()


def blockwise(iterable, step=1, start=0):
    """
    Blockwise iterator with blocksize equal to step parameter.
    """
    if np.sign(step) == -1:
        logger.warn(
            "Blockwise will lose first value on negative step, "
            "because of Numpy array's indexing bug."
        )

    def reset(start=start):
        """
        Reset the blockwise iterator.
        """
        return pairwise(count_step(start, step))

    blocks = reset()
    while True:
        try:
            indices = blocks.next()
            block = iterable[slice(*np.append(indices, np.sign(step)))]
        except StopIteration:
            break
        except GeneratorExit:
            logger.debug("Blockwise generator exit.")
            break

        if len(block) > 0:
            val = (yield block)  # sending values into generator with send
            while val is not None:
                if val == 'current':
                    val = (yield indices)
                if val == 'reset':
                    blocks = reset()
                    indices = (start, step)
                    val = (yield True)
        else:
            raise StopIteration

    del block, blocks, indices
