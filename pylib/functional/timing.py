#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Math &c.
import numpy as np


class Sampler(object):
    rate = 44100


def stime(start, end, rate=Sampler.rate):
    return np.linspace(start, end, (end-start)*rate, endpoint=False)