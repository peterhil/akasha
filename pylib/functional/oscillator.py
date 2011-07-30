#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np

# My modules
from timing import Sampler, stime

# Utils
from utils.math import *
from utils.graphing import *
from utils.audio import play, write

# Functional
import itertools as itr
import functools as fun
from xoltar import lazy
from xoltar.functional import *

# Settings
np.set_printoptions(precision=16, suppress=True)

def accumulator(n):
    """Function object using closure, see:
    http://en.wikipedia.org/wiki/Function_object#In_Python"""
    def inc(x):
        inc.n += x
        return inc.n
    inc.n = n
    return inc

def osc(freq):
    def osc(times):
        osc.gen = 1j * 2 * np.pi
        return np.exp( osc.gen * osc.freq * (times % (1.0 / osc.freq)) )
    osc.freq = freq
    def ratio():
        return osc.freq / float(Sampler.rate)
    osc.ratio = ratio
    return osc

def exp(rate):
    def exp(times):
        return np.exp( exp.rate * times )
    exp.rate = rate
    return exp
