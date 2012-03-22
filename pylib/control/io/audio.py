#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import string
import numpy as np
import os

from scipy.signal import hilbert
from scikits import audiolab
from scikits.audiolab import Format, Sndfile, available_file_formats, available_encodings

from ...timing import Sampler, time_slice


defaults = {
    'type': 'aiff',
    'encoding': 'pcm16',
    'endianness': 'file'
}

default_format = Format(**defaults)

def get_format(*args, **kwargs):
    """
    Get a audiolab.Format object from a string, or from the parameters
    accepted by Format(type=wav, encoding=pcm16, endianness=file)
    """
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
    """
    Play out a sound.
    """
    time = time_slice(dur, start, time)
    if isinstance(sndobj[0], np.floating): axis = 'real'
    audiolab.play(getattr(sndobj[time], axis), fs)

def write(sndobj, filename='test_sound', axis='imag', fmt=Format(**defaults),
          fs=Sampler.rate, dur=5.0, start=0, time=False,
          sdir='../../Sounds/2010_Python_Resonance/'):
    """
    Write a sound file.
    """
    fmt = get_format(fmt)
    filename = sdir + filename +'_'+ axis +'.'+ fmt.file_format

    time = time_slice(dur, start, time)
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
    """
    Read a sound file in.
    """
    # TODO:
    # - Factor out the real->complex conversion out.
    # - If fs or enc differs from default do some conversion?

    if filename[0] != '/':
        filename = sdir + filename # Relative path

    format = os.path.splitext(filename)[1][1:]
    check_format(format)
    time = time_slice(dur, start, time)

    # Get and call appropriate reader function
    func = getattr(audiolab, format + 'read')
    (data, fs, enc) = func(filename, last=time.stop, first=time.start)

    if data.ndim > 1:
        data = data.transpose()[-1]  # Make mono, take left [0] or right [-1] channel.

    if complex:
        return hilbert(data)
    else:
        return data

write.__doc__ += "Available file formats are: %s." % (', '.join([f for f in available_file_formats()]))
read.__doc__ += "Available file formats are: %s." % (', '.join([f for f in available_file_formats()]))

def check_format(format):
    """Checks that a requested format is available (in libsndfile)."""
    available = available_file_formats()
    if format not in available:
        raise ValueError("File format '%s' not available. Try one of: %s" % (format, available))

def check_encoding(format):
    pass

