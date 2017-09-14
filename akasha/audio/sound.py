#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Generic sound object group containers.
"""

import numpy as np

from numbers import Number
from scikits import samplerate as src

from akasha import dsp

from akasha.audio.frequency import FrequencyRatioMixin, Frequency
from akasha.audio.generators import Generator
from akasha.funct import blockwise
from akasha.timing import sampler
from akasha.utils.log import logger


class Pcm(FrequencyRatioMixin, Generator):
    """
    Playable PCM (pulse-code modulated aka sampled) sound.
    """
    def __init__(self, snd, base = 441):
        super(self.__class__, self).__init__()
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
                len(self.snd) * (self.base_freq.ratio / self.frequency.ratio)
            )))

    @staticmethod
    def resample(signal, ratio, window='linear'):
        """
        Resample the PCM sound with a new normalized frequency (ratio).
        """
        logger.info(
            "Resample at {0} ({1:.3f}). Hilbert transform may cause clipping!"
            .format(ratio, float(ratio))
        )
        orig_state = sampler.paused
        sampler.paused = True
        out = dsp.signal.hilbert(src.resample(signal.real, float(ratio), window)).astype(np.complex128)
        sampler.paused = orig_state
        return out

    @staticmethod
    def sc_resample(signal, ratio, window='blackman'):
        """
        Resample the PCM sound with a new normalized frequency (ratio).
        Uses scipy.signal.resample.
        """
        # TODO: This sounds better than scikits.samplerate, but try
        # if something else is faster with complex samples!
        #
        # Note about scipy.signal.resample: t : array_like, optional
        # If t given, it's assumed to be the sample positions associated with the signal data in x
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
        return dsp.signal.resample(signal, int(round(ratio * len(signal))), window = window)

    def resample_at_freq(self, items=None):
        """
        Resample and get items at self frequency.
        """
        if items is None:
            items = slice(0, len(self))
        ratio = (self.base_freq.ratio / self.frequency.ratio)
        if ratio == 0:
            return np.array([0j])
        elif ratio == 1:
            return self.snd[items]
        else:
            if isinstance(items, slice) and items.stop >= len(self):
                logger.warn("Normalising {0} for length {1}".format(items, len(self)))
                items = slice(items.start, min(items.stop, len(self), items.step))
            # return self.resample(self.snd[items], ratio)
            return self.sc_resample(self.snd[items], ratio)

    def sample(self, items):
        """
        Sample the pcm sampled sound signal.
        """
        return self.resample_at_freq(items)


class Sound(Generator):
    """
    Collection or composition of sound objects.
    """
    def __init__(self, *args):
        super(self.__class__, self).__init__()
        self.sounds = {}
        for s in args:
            self.add(s)

    def sample(self, items):
        """
        Sample all sound objects.
        """
        if isinstance(items, Number):
            # FIXME should return scalar, not array!
            start = int(items)
            stop = start + 1
        elif isinstance(items, np.ndarray):
            start = 0
            stop = len(items)
        else:
            start = items.start or 0
            stop = items.stop
        sl = (start, stop)

        sound = np.zeros((stop - start), dtype=complex)
        for sl in self.sounds:
            #print "Slice start %s, stop %s" % sl
            for sndobj in self.sounds[sl]:
                #print "Sound object %s" % sndobj
                sound += sndobj[items]
        return sound / max(len(self.sounds), 1.0)

    def add(self, sndobj, start=0, dur=None):
        """
        Add a sound object.
        """
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
