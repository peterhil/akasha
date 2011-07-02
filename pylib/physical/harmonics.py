#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

# Math
import numpy as np
from fractions import Fraction

# My modules
from physical.envelope import Exponential
from physical.oscillator import Osc
from physical.generators import Generator

# Utils
from utils import play, wavwrite

# np.set_printoptions(precision=4, suppress=True)

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

class Harmonic(object, Generator):
    """Harmonical overtones"""

    def __init__(self, func=lambda x: 1+x, n=8):
        # Set overtones
        self.func = func
        self.limit = n
        if n <= 20:
            self.overtones = np.array(map(func, np.arange(0, n)), dtype=np.float32)
        else:
            # numpy.apply_along_axis is faster than map for larger n
            self.overtones = np.apply_along_axis(func, 0, np.arange(0, n, dtype=np.float32))

    def __call__(self, freq):
        self.freq = freq
        return self

    def sample(self, iter):
        oscs = Osc.freq(self.freq) * self.overtones
        oscs = np.ma.masked_array(oscs, np.equal(oscs, Osc(0, 1)), None).compressed()
        # oscs = filter(lambda x: x!=Osc(0,1), oscs)  # Quick hack to prevent problems with numpy broadcasting and new style classes
        frames = np.zeros(len(iter), dtype=complex)
        for o in oscs:
            # e = Exponential(0, amp=float(self.freq)/o.frequency*float(self.freq)) # square waves
            # e = Exponential(0, amp=float(self.freq)**2/o.frequency**2*float(self.freq)) # triangle waves
            e = Exponential(-o.frequency/100.0) # sine waves
            frames += o[iter] * e[iter]
        return frames / max( abs(max(frames)), len(oscs), 1.0 )

    # def __repr__(self):
    #     object_str = u''
    #     for obj in self.sounds:
    #         objects_str += repr(obj)
    #     return "Harmonic(%s)" % (object_str)
