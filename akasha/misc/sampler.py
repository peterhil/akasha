#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Many ways to use properties
"""

from akasha.misc.prop import prop
from utils.borg import Borg


class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


class sampler_(object):
    __metaclass__ = Singleton

    def __init__(self, rate=44100):
        self.rate = rate


class Rate(Borg):
    def __init__(self):
        Borg.__init__(self)


class samplerate(object):
    """
    A descriptor object for sample rate.
    TODO: also handle sending resampling calls to objects?
    """
    __metaclass__ = Singleton
    default_rate = 44100

    def __init__(self, value=None):
        self.rate = value

    def __get__(self, obj, objtype):
        print "Self: %s, Obj: %s, Object type: %s" % (self.rate, obj, objtype)
        return self.rate or self.__set__(objtype, self.default_rate)

    def __set__(self, obj, val):
        print "Setting sampling rate %s for %s" % (val, obj)
        self.rate = val


class sampler:
    rate = 44100

    #@prop
    def rating():
        """ Sample rate"""
        return {'fget': lambda self: getattr(self, 'rate')}
    prop(rating)

    @classmethod
    def _get_tuning(cls):
        print "Here"
        if type(cls.rate) == property:
            cls.rate = 44100
        return cls.rate

    @classmethod
    def _set_tuning(cls, rate):
        cls.rate = rate
    tuning = property(_get_tuning, _set_tuning)
