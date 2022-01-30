#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Oop design patterns module
"""


class Singleton:
    """
    Create a singleton.
    Use as a mixin to make all instances of a class singletons.
    """

    def __new__(cls, *args, **kwds):
        """
        >>> s = Singleton()
        >>> p = Singleton()
        >>> id(s) == id(p)
        True
        """
        self = "__self__"
        if not hasattr(cls, self):
            instance = object.__new__(cls)
            instance.init(*args, **kwds)
            setattr(cls, self, instance)
        return getattr(cls, self)

    def init(self, *args, **kwds):
        """Disable init"""
        pass
