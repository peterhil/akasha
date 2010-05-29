#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scikits import audiolab
from timing import Sampler

def play(sndobj, time=slice(0, Sampler.rate), axis='imag', fs=Sampler.rate):
    # time = time or slice(0, Sampler.rate)
    # fs = fs or Sampler.rate
    audiolab.play(getattr(sndobj[time], axis), fs)

def wavwrite(*arg, **kwargs):
    audiolab.wavwrite(*args, **kwargs)
# def wavwrite(sound, dir='../data/Sounds/', fs=Sampler.rate, enc='pcm16')
#     filename
#     audiolab.wavwrite(sound, )