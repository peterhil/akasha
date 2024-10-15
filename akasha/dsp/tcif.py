"""
Time corrected instantaneous frequency (TCIF) algorithms module.
"""

import numpy as np
import scipy as sc

from akasha.dsp.fft import stft
from akasha.math.functions import pad, pi2, power_limit
from akasha.timing import sampler


__all__ = ['tcif']


def tcif(signal, window_size=1901, hop=None, method='kodera'):
    """
    TCIF algorithm using the finite difference approximations by Kodera et al.
    """
    if hop is None:
        hop = window_size // 2
    # Find the next power of two for n_fft
    n_fft = int(power_limit(window_size, base=2, rounding=np.ceil))

    # Step 1 and 2, build the matrices and do three stfts

    # TODO Maybe use np.roll?
    delayed_signal = pad(signal[1:-1], index=0, count=1, value=0)
    # TODO Try other windows than hamming
    args = dict(
        n_fft=n_fft,
        frame_size=window_size,
        hop=hop,
        window=sc.signal.hamming,
        roll=False,
        normalize=True,
    )
    s = stft(signal, **args)
    s_delayed = stft(delayed_signal, **args)
    # TODO Check that frequencies are rolled up by one!
    s_freqdel = np.roll(s, 1, axis=0)

    # Step 3, compute channelized instantaneous frequency and
    # local group delay

    if method == 'kodera':
        abs_delayed = np.abs(s_delayed) - np.abs(s)
        abs_freqdel = np.abs(s_freqdel) - np.abs(s)
        cif = (-sampler.rate / pi2) * np.mod(abs_delayed, pi2)
        lgd = (n_fft / (pi2 * sampler.rate)) * np.mod(abs_freqdel, pi2)
    elif method == 'nelson':
        raise NotImplementedError('Nelson method is not implemented yet')
    else:
        raise NotImplementedError('Given method is not implemented')

    return (cif, lgd)  # TODO Implement plotting (in a separate function?)
