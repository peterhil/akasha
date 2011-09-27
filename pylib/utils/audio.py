#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import string
import os

import numpy as np
from scipy.signal import hilbert
from scikits import audiolab
from scikits.audiolab import Format, Sndfile, available_file_formats, available_encodings

from timing import Sampler

defaults = {
    'type': 'aiff',
    'encoding': 'pcm16',
    'endianness': 'file'
    }

default_format = Format(**defaults)

def get_format(*args, **kwargs):
    """Get a Format object from a string, or parameters accepted by Format(type=wav, encoding=pcm16, endianness=file)"""
    if (args and isinstance(args[0], Format)):
        res = args[0]
    else:
        params = defaults.copy()
        for i in xrange(len(args)):
            params[['type', 'encoding', 'endianness'][i]] = args[i]
        params.update(kwargs)
        res = Format(**params)
    return res

# Audiolab read, write and play

def play(sndobj, axis='imag', fs=Sampler.rate, dur=5.0, start=0, time=False):
    time = _slice_from(dur, start, time)
    audiolab.play(getattr(sndobj[time], axis), fs)

def write(sndobj, filename='test_sound', axis='imag', fmt=Format(**defaults),
          fs=Sampler.rate, dur=5.0, start=0, time=False,
          sdir='../../Sounds/2010_Python_Resonance/'):

    fmt = get_format(fmt)
    filename = sdir + filename +'_'+ axis +'.'+ fmt.file_format

    time = _slice_from(dur, start, time)
    data = getattr(sndobj[time], axis)

    if np.ndim(data) <= 1:
        n_channels = 1
    elif np.ndim(data) == 2:
        n_channels = data.shape[1]
    else:
        RuntimeError("Only rank 0, 1, and 2 arrays supported as audio data")

    hdl = Sndfile(filename, 'w', fmt, n_channels, int(fs))
    try:
        hdl.write_frames(data)
    finally:
        hdl.close()


def read(filename, dur=5.0, start=0, time=False, fs=Sampler.rate, complex=True,
         sdir='../../Sounds/_Music samples/', *args, **kwargs):
    """Reading function. Useful for doing some analysis."""

    # Process:
    # Get audiofile in
    # Process
    # Return complex audio data

    # Operation:
    # - examine file extension to determine type

    # TODO:
    # - Factor out the real->complex conversion out.

    if filename[0] != '/':    # Relative path
        filename = sdir + filename

    format = os.path.splitext(filename)[1][1:]
    check_format(format)
    time = _slice_from(dur, start, time)

    # Get and call appropriate reader function
    func = getattr(audiolab, format + 'read')

    # Read data
    (data, fs, enc) = func(filename, last=time.stop, first=time.start)   # returns (data, fs, enc)

    # TODO if fs or enc differs from default do some conversion?
    if data.ndim > 1:
        data = data.transpose()[-1]  # Make mono, take left [0] or right [-1] channel.

    # Complex or real samples?
    if complex:
        data = hilbert(data)

    return data

def check_format(format):
    """Checks that a requested format is available (in libsndfile)."""
    available = available_file_formats()
    if format not in available:
        raise ValueError("File format '%s' not available. Try one of: %s" % (format, available))

def check_encoding(format):
    pass

def _slice_from(dur, start, time=False):
    """Use a time slice argument or the provided attributes 'dur' and 'start' to
    construct a time slice object."""
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))
    if not isinstance(time, slice):
        raise TypeError("Expected a %s for 'time' argument, got %s." % (slice, type(time)))
    return time

