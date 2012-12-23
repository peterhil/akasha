#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numbers import Number
from scipy import signal as dsp
from scikits import samplerate as src

from akasha.audio.frequency import FrequencyRatioMixin, Frequency
from akasha.audio.generators import Generator
from akasha.funct import blockwise
from akasha.timing import sampler

from akasha.utils.decorators import memoized
from akasha.utils.log import logger


class Pcm(FrequencyRatioMixin, Generator):
    """
    A playable sampled (pcm) sound.
    """
    def __init__(self, snd, base=1):
        self._hz = Frequency(base)
        self.base_freq = Frequency(base)
        self.snd = snd

    def __iter__(self):
        return blockwise(self.resample_at_freq(), sampler.blocksize())

    def __len__(self):
        if self.frequency == 0:
            return 1
        else:
            return int(np.floor(float(
                len(self.snd) * (self.base_freq.ratio / self.frequency.ratio))))

    @memoized
    def resample(self, ratio, window='linear'):
        logger.info(
            "Resample at {0} ({1:.3f}). Hilbert transform may cause clipping!".format(
                ratio, float(ratio)))
        orig_state = sampler.paused
        sampler.paused = True
        out = dsp.hilbert(src.resample(self.snd.real, float(ratio), window)).astype(np.complex128)
        sampler.paused = orig_state
        return out
        #return src.resample(self.snd.real, float(ratio), window, verbose=True).astype(np.float64)

    @memoized
    def sc_resample(self, ratio, window='blackman'):
        # TODO: This sounds better than scikits.samplerate, but try
        # if something else is faster with complex samples!
        #
        # Note about scipy.signal.resample: t : array_like, optional
        # If t given, it's assumed to be the sample positions associated with the signal data in x
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
        return dsp.resample(self.snd, len(self))

    def resample_at_freq(self, iter=None):
        if iter is None:
            iter = slice(0, len(self))
        ratio = (self.base_freq.ratio / self.frequency.ratio)
        if ratio == 0:
            return np.array([0j])
        elif ratio == 1:
            return self.snd[iter]
        else:
            if isinstance(iter, slice) and iter.stop >= len(self):
                logger.warn("Normalising {0} for length {1}".format(iter, len(self)))
                iter = slice(iter.start, min(iter.stop, len(self), iter.step))
            return self.resample(ratio)[iter]
            #return self.sc_resample(ratio)[iter]

    def sample(self, iter):
        #logger.debug(__name__ + " sample("+str(self)+"): " + str(iter))
        return self.resample_at_freq(iter)


class Group(FrequencyRatioMixin, Generator):
    """A group of sound objects."""

    def __init__(self, *args):
        self.sounds = np.array(*args, dtype=object)
        # TODO handle zero-frequencies and non-periodic sounds:
        self.frequency = np.min(np.ma.masked_equal(args, 0).compressed())


class Sound(Generator):
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
        return sound / max(len(self), 1.0)

    def add(self, sndobj, start=0, dur=None):
        """Add a new sndobj to self."""
        if dur:
            end = start + dur
        elif hasattr(sndobj, "len"):
            end = start + len(sndobj)
        else:
            end = None

        sl = (start, end)

        if (sl in self.sounds):
            self.sounds[sl].append(sndobj)
        else:
            self.sounds[sl] = [sndobj]
        return self

