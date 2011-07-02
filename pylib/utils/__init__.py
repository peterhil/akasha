#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import string
import os
from itertools import tee, izip

import numpy as np
from scipy.signal import hilbert
from scikits import audiolab

from timing import Sampler

# itertools recipes -- http://docs.python.org/library/itertools.html#recipes
def take(n, iterable):
    "Return first n items of the iterable as a Numpy Array"
    return np.fromiter(islice(iterable, n), dtype=iterable.dtype)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# Audiolab read, write and play

available_formats = set(map(lambda s: string.replace(s, 'write', ''), audiolab.__all__)) & set(audiolab.available_file_formats())

def play(sndobj, axis='imag', fs=Sampler.rate, dur=5.0, start=0, time=False):
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))
    audiolab.play(getattr(sndobj[time], axis), fs)

def write(sndobj, filename='test_sound', axis='imag', format='aiff', enc='pcm16',
          fs=Sampler.rate, dur=5.0, start=0, time=False,
          sdir='../../Sounds/2010_Python_Resonance/', *args, **kwargs):

    # Check that format is available
    if format not in available_formats:
        raise ValueError("File format '%s' not available. Try one of: %s" % (format, list(available_formats)))

    # Use time (=slice obj) OR the provided attributes dur and start
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))

    # Get and call appropriate writer function
    func = getattr(audiolab, format + 'write')
    return func(getattr(sndobj[time], axis), sdir + filename +'_'+ axis +'.'+ format, fs, enc)

def read(filename, fs=Sampler.rate, dur=5.0, start=0, time=False, complex=True,
         sdir='../../Sounds/_Music samples/', *args, **kwargs):
    """Reading function. Useful for doing some analysis. Audiolab has the same read as write functions!"""

    if filename[0] != '/':    # Relative path
        filename = sdir + filename

    format = os.path.splitext(filename)[1][1:]

    # Check that format is available
    if format not in available_formats:
        raise ValueError("File format '%s' not available. Try one of: %s" % (format, list(available_formats)))

    # Use time (=slice obj) OR the provided attributes dur and start
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))

    # Get and call appropriate reader function
    func = getattr(audiolab, format + 'read')

    # Read data
    (data, fs, enc) = func(filename, last=time.stop, first=start)   # returns (data, fs, enc)
    # TODO if fs or enc differs from default do some conversion?
    if data.ndim > 1:
        data = data.transpose()[-1]  # Make mono, take left [0] or right [-1] channel.

    # Complex or real samples?
    if complex:
        data = hilbert(data)

    return data
