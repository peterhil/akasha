"""
Pcm sound sample module
"""

import numpy as np

from scikits import samplerate as src

from akasha import dsp

from akasha.audio.frequency import FrequencyRatioMixin, Frequency
from akasha.audio.generators import Generator
from akasha.funct.itertools import blockwise
from akasha.timing import sampler
from akasha.utils.log import logger
from akasha.utils.python import _super


# FIXME Make this work again at some point or remove
# TODO Try using CZT, multiply and ICZT to make resampling faster, but
# not so accurate
# TODO Rewrite if and when I manage to make the vector audio samples
# based on clothoid curve or complex exponential spiral fitting
class Resample(FrequencyRatioMixin, Generator):
    """PCM (pulse-code modulated aka sampled) sound sample that is
    resampled when played.
    """

    def __init__(self, snd, base=441):
        _super(self).__init__()
        self._hz = Frequency(base)
        self.base_freq = Frequency(base)
        self.snd = snd

    def __iter__(self):
        return blockwise(self.resample_at_freq(), sampler.blocksize())

    def __len__(self):
        if self.frequency == 0:
            return 1
        ratio = float(self.base_freq.ratio / self.frequency.ratio)

        return int(np.ceil(len(self.snd) * ratio))

    @staticmethod
    def resample(signal, ratio, window='linear'):
        """
        Resample the PCM sound with a new normalized frequency (ratio).
        """
        logger.info(
            'Resample at %r (%.3f).' 'Hilbert transform may cause clipping!',
            ratio,
            float(ratio),
        )
        orig_state = sampler.paused
        sampler.paused = True
        out = dsp.signal.hilbert(
            src.resample(signal.real, float(ratio), window)
        ).astype(np.complex128)
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
        return dsp.signal.resample(
            signal,
            int(round(ratio * len(signal))),
            window=window,
        )

    def resample_at_freq(self, items=None):
        """
        Resample and get items at self frequency.
        """
        if items is None:
            items = slice(0, len(self))
        ratio = self.base_freq.ratio / self.frequency.ratio

        if ratio == 0:
            return np.array([0j])
        if ratio == 1:
            return self.snd[items]

        if isinstance(items, slice) and items.stop >= len(self):
            logger.warning(
                'Normalising %r for length %d',
                items,
                len(items),
            )
            stop = min(items.stop, len(self))
            items = slice(items.start, stop, items.step)

        # return self.resample(self.snd[items], ratio)
        return self.sc_resample(self.snd[items], ratio)

    def sample(self, frames):
        """
        Sample the pcm sampled sound signal.
        """
        return self.resample_at_freq(frames)
