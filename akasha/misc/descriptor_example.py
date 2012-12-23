#!/usr/bin/env python
# -*- coding: utf-8 -*-


class RevealAccess(object):
    """A data descriptor that sets and returns values
       normally and prints a message logging their access.
    """

    def __init__(self, initval=None, name='var'):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print 'Retrieving', self.name, self.val
        return self.val

    def __set__(self, obj, value):
        print 'Updating' , self.name, value
        self.val = value


class samplerate(object):
    """A descriptor object for sample rate.
       TODO: also handle sending resampling calls to objects?
    """
    # default_rate = 44100

    def __init__(self, value=None):
        self.val = value

    def __get__(self, obj, objtype):
        print "Self: %s, Obj: %s, Object type: %s" % (self.val, obj, objtype)
        return self.val #or self.__set__(objtype, self.default_rate)

    def __set__(self, obj, val):
        print "Setting sampling rate %s for %s" % (val, obj)
        self.val = val


class MyClass(object):
    x = RevealAccess(10, 'var "x"')
    y = RevealAccess(5, 'var "y"')


# >>> m = MyClass()
# >>> m.x
# Retrieving var "x"
# 10
# >>> m.x = 20
# Updating var "x"
# >>> m.x
# Retrieving var "x"
# 20
# >>> m.y
# 5


if __name__ == '__main__':
    m = MyClass()
    m.x
    m.x = 20
    m.x
    m.x = 40
    m.y

