#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scikits import audiolab
from timing import Sampler

available_formats = set(map(lambda s: string.replace(s, 'write', ''), audiolab.__all__)) & set(audiolab.available_file_formats())

def play(sndobj, axis='imag', fs=Sampler.rate, dur=1.0, start=0, time=False):
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))
    audiolab.play(getattr(sndobj[time], axis), fs)

def write(sndobj, filename='test_sound', axis='imag', format='aiff', enc='pcm16', 
          fs=Sampler.rate, dur=1.0, start=0, time=False, 
          sdir='../../Sounds/2010_Python_Resonance/', *args, **kwargs):
    
    # Check that format is available
    if format not in available_formats:
        raise ValueError("File format '%s' not available. Try one of: %s" % (format, list(available_formats)))
    
    # Use time (=slice obj) OR the provided attributes dur and start
    time = time or slice(int(round(0 + start)), int(round(dur * Sampler.rate + start)))
    
    # Get and call appropriate writer function
    func = getattr(audiolab, format + 'write')
    func(getattr(sndobj[time], axis), sdir + filename +'_'+ axis +'.'+ format, fs, enc)

# TODO: Write reading function later, when doing some analysis. Audiolab has the same read as write functions!