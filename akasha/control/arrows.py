#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

from exceptions import TypeError, AttributeError
from numbers import Number

from .monads import fid, decorator_with_args


class Arrow(object):
    def __init__(self, func):
        if hasattr(func, 'signature'):
            self.func = func
        else:
            # Could do:
            # self.func = lift(func, object, object)
            raise NotImplementedError
    
    # Make these properties excplit
    @property
    def signature(self):
        return self.func.signature
    
    @property
    def accepts(self):
        return self.func.accepts
    
    @property
    def returns(self):
        return self.func.returns
    
    def __getattr__(self, name):
        if (self.__dict__.has_key(name)):
            return self.__dict__[name]
        else:
            try:
                return self.func.__getattribute__(name)
            except AttributeError, e:
                raise AttributeError(e)

    def __call__(self, *args, **kargs):
        return self.func(*args, **kargs)

    def bind(self, bindee):
        #raise NotImplementedError
        f = self.func
        g = arr(bindee)
        if (g.returns == f.accepts):
            o = lift(compose(f, g), g.accepts, f.returns)
            return Arrow(o)
        else:
            msg_arrows = (f, f.signature, g, g.signature)
            raise TypeError("Incompatible types. Can't bind arrow a {0} of type {1} with arrow b {2} of type {3}".format(msg_arrows))

    def __rshift__(self, bindee):
        return self.bind(bindee)


def arr(func):
    if isinstance(func, Arrow):
        return func
    else:
        return Arrow(func)


def mcompose(*fns):
    return reduce(compose, fns)

def npcompose(f, g):
    def fog(*x):
        return f(g(np.atleast_1d(*x)))
    fog.__name__ = f.__name__ + '_o_' + g.__name__
    return fog

def compose(f, g, unpack=False):
    def fog(*x):
        return f(g(*[x]))
    fog.__name__ = f.__name__ + '_o_' + g.__name__
    return fog


def decoratorFunctionWithArguments(arg1, arg2, arg3):
    def wrap(f):
        print "Inside wrap()"
        def wrapped_f(*args):
            print "Inside wrapped_f()"
            print "Decorator arguments:", arg1, arg2, arg3
            f(*args)
            print "After f(*args)"
        return wrapped_f
    return wrap

def check_args(func, *args, **kargs):
    ### TODO handle kwargs
    print "check_args:", func, args, kargs
    print func.__dict__
    for (arg, arg_type) in zip(args, func.accepts):
        print arg, arg_type
        if not issubclass(type(arg), arg_type):
            raise TypeError("Argument {0} has wrong type {1}, expected {2}.".format(arg, type(arg), arg_type))
    res = func(*args)
    if not issubclass(type(res), func.returns):
        raise TypeError("Function {0} returns wrong type {1}, expected {2}.".format(
            func.__name__,
            type(res),
            func.returns
            ))
    else:
        return res

def lift(func, accepts, returns):
    @ftype(accepts, returns)
    def lifted(*args, **kargs):
        return func(*args, **kargs)
    lifted.__name__ = func.__name__
    return lifted

def ftype(accepts, returns):
    """Decorator that declares and checks function type.
    A very simple implementation of Haskell like type system to ensure correct composition of functions.
    """
    def make_decorator(func):
        print "Inside make_decorator:", func, accepts, returns
        func.accepts = accepts
        func.returns = returns
        func.signature = {'accepts': accepts, 'returns': returns}
            
        def decorated(*args, **kargs):
            print "Inside decorated:", args, kargs
            return check_args(func, *args, **kargs)
        
        decorated.__name__  = func.__name__
        decorated.__doc__   = func.__doc__
        decorated.accepts = accepts
        decorated.returns = returns
        decorated.signature = {'accepts': accepts, 'returns': returns}
            
        print "Func signature", func.signature
        return decorated
    
    return make_decorator


class typed(object):
    """Decorator that declares and checks function type.
    A very simple implementation of Haskell like type system to ensure correct composition of functions.
    """
    def __init__(self, accepts, returns):
        print "inside init():", self, accepts, returns
        self.signature = {'accepts': accepts, 'returns': returns}
        print self.signature

    @property
    def accepts(self):
        return self.signature['accepts']

    @property
    def returns(self):
        return self.signature['returns']

    def check_args(self, *args):
        print "inside check_args:", self, args
        for (arg, arg_type) in zip(args, self.accepts):
            print arg, arg_type
            if not issubclass(type(arg), arg_type):
                    raise TypeError("Argument {0} has wrong type {1}, expected {2}.".format(arg, type(arg), arg_type))
        res = self.func(*args)
        if not issubclass(type(res), self.returns):
            raise TypeError("Function {0} returns wrong type {1}, expected {2}.".format(
                self.func.func_name,
                type(res),
                self.returns
                ))
        else:
            return res

    def __call__(self, func, *args):
        self.func = func
        print "inside call(): args", self, func, args
        def wrap(*args):
            print "inside wrap()", self, func, args
            def decorated(*args):
                print "inside decorated()", self, func, args
                return self.check_args(*args)
            return decorated
        wrap.__name__ = func.__name__
        return wrap(*args)
