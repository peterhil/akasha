#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from numbers import Number
from scipy import signal as dsp
from scikits import samplerate as src

from .frequency import FrequencyRatioMixin, Frequency
from .generators import Generator
from ..funct import blockwise, blockwise2
from ..timing import sampler

from ..utils.decorators import memoized
from ..utils.log import logger
from ..utils.math import *


class Pcm(FrequencyRatioMixin, Generator, object):
    """
    A playable sampled (pcm) sound.
    """
    def __init__(self, snd, base=1):
        self._hz = Frequency(base)
        self.base_freq = Frequency(base)
        self.snd = snd

    def __iter__(self):
        return blockwise2(self.resample_at_freq(), sampler.blocksize())

    def __len__(self):
        return max(int(round(len(self.snd) * (self.base_freq.ratio / self.frequency.ratio))), 1)

    @memoized
    def resample(self, ratio, window='linear'):
        ratio = float(ratio)
        if ratio == 0.0:
            logger.info("Resampling at 0.0: returning [0j]!")
            return np.array([0j])
        if ratio == 1.0:
            logger.info("Resampling at 1.0: returning self.snd!")
            return self.snd
        else:
            logger.info("Resampling at %s. Note that hilbert transform may cause clipping!" % ratio)
            return dsp.hilbert(src.resample(self.snd.real, ratio, window)).astype(np.complex128)
            #return dsp.hilbert(src.resample(self.snd, ratio, window, verbose=False).astype(np.float64))

    @memoized
    def sc_resample(self, ratio, window='blackman'):
        # TODO: This sounds better than scikits.samplerate, but try if something else is faster with complex samples!
        #
        # Note about scipy.signal.resample: t : array_like, optional
        # If t is given, it is assumed to be the sample positions associated with the signal data in x.
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
        return dsp.resample(self.snd, len(self))

    def resample_at_freq(self):
        ratio = float(self.base_freq.ratio / self.frequency.ratio)
        logger.debug(__name__ + " resample_at_freq() at ratio: " + str(ratio) + " self: " + str(self))
        return self.resample(self, ratio)

    def sample(self, iter):
        logger.debug(__name__ + " sample("+str(self)+"): " + str(iter))
        return self.resample_at_freq()[iter]


class Group(FrequencyRatioMixin, Generator, object):
    """A group of sound objects."""

    def __init__(self, *args):
        self.sounds = np.array(*args, dtype=object)
        # TODO handle zero-frequencies and non-periodic sounds:
        self.frequency = np.min(np.ma.masked_equal(args, 0).compressed())


class Sound(Generator, object):
    """A group of sound objects."""

    def __init__(self, *args):
        self.sounds = {}
        for s in args:
            self.add(s)

    def sample(self, iter):
        """Pass parameters to all sound objects and update states."""
        if isinstance(iter, Number):
            # FIXME should return scalar, not array!
            start = int(iter)
            stop = start + 1
        elif isinstance(iter, np.ndarray):
            start = 0
            stop = len(iter)
        else:
            start = iter.start or 0
            stop = iter.stop
        sl = (start, stop)

        sound = np.zeros((stop - start), dtype=complex)
        for sl in self.sounds:
            #print "Slice start %s, stop %s" % sl
            for sndobj in self.sounds[sl]:
                #print "Sound object %s" % sndobj
                sound += sndobj[iter]
        return sound / max( len(self), 1.0 )

    def add(self, sndobj, start=0, dur=None):
        """Add a new sndobj to self."""
        if dur:
            end = start + dur
        elif hasattr(sndobj, "len"):
            end = start + len(sndobj)
        else:
            end = None

        sl = (start, end)

        if (self.sounds.has_key(sl)):
            self.sounds[sl].append(sndobj)
        else:
            self.sounds[sl] = [sndobj]
        return self

