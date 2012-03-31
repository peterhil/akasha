#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


class sampler(object):
    rate = 44100

def samples(start, end, rate=sampler.rate):
    return np.arange(int(np.ceil(start * rate)), int(round(np.floor(end * rate))))

def times(start, end, rate=sampler.rate):
    return samples(start, end, rate) / float(rate)
