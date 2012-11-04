#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import collections
import numpy as np

from itertools import *

from ..utils.log import logger


# function objects - or a kund of function composition?

def _findvars(*funcs):
    import re
    idents = re.compile(r'([a-zA-Z_]\w*)') # (letter|"_") (letter | digit | "_")*
    nfuncs = {}
    for func in funcs:
        vars = tuple(set(idents.findall(func)))
        if len(vars) == 1:
            vars = vars[0]
        nfuncs.update([[vars, func]])
    return nfuncs

class fn(object):

    def __init__(self, *funcs, **nfuncs):
        self.__dict__ = nfuncs

    def __call__(self, *args, **kwargs):
        for vars, func in self.__dict__:
            func(*args, **kwargs)

    def __repr__(self):
        return repr(self.__dict__)


# itertools recipes -- http://docs.python.org/library/itertools.html#recipes

def take(n, iterable):
    "Return first n items of the iterable as a Numpy Array"
    return np.fromiter(islice(iterable, n), dtype=iterable.dtype)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

# Other iterators and generators

def count_step(start=0, step=1):
	g = (start + step * i for i in count())
	yield g.next()
	while True:
		yield g.next()

def blockwise(iterable, step=1, start=0):
	if np.sign(step) == -1:
		logger.warn("Blockwise will lose first value on negative step, because Numpy array's indexing bug.")

	def reset(start=start):
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
			val = (yield block) # sending values into generator with send
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

