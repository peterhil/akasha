#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decorators module
"""

from akasha.utils.log import logger


class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            self.cache[args] = value = self.func(*args)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            logger.warning("Arguments '%s' not memoized for function %s!" % (args, self.func))
            return self.func(*args)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__
