#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# Math &c.
import numpy as np

# My modules
from timing import Sampler
from utils.graphing import *
import utils

# Settings
np.set_printoptions(precision=16, suppress=True)


# In [495]: np.linspace(0,1,44100,endpoint=False)
# Out[495]:
# array([ 0.        ,  0.00002268,  0.00004535,  0.00006803,  0.0000907 ,
#         0.00011338,  0.00013605,  0.00015873,  0.00018141,  0.00020408,
#        ...,  0.99977324,  0.99979592,  0.99981859,  0.99984127,
#         0.99986395,  0.99988662,  0.9999093 ,  0.99993197,  0.99995465,
#         0.99997732])

def stime(start, end, rate=Sampler.rate):
    return np.linspace(start, end, (end-start)*rate, endpoint=False)


class Generator:
    # Could be new style class, but this causes many problems because numpy
    # uses obj[0] (arrays first element) to determine it's type and then
    # automatically broadcasts oscs to their complex samples!
    # This could maybe be prevented by using custom __get* methods or descriptors.

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)

    def play(self, *args, **kwargs):
        utils.play(self, *args, **kwargs)


class Osc(object, Generator):
    """Oscillator class"""

    def __init__(self, freq):
        self.freq = freq

    def sample(self, slice)
