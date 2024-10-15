#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Fast fourier transforms.
"""

import numpy as np
import scipy as sc
import types
import pylab

from akasha.dsp.window import sliding_window
from akasha.dsp.z_transform import czt
from akasha.timing import sampler
from akasha.utils.array import is_sequence


def stft(
    signal,
    n_fft=2048,
    frame_size=None,
    hop=None,
    window=np.hamming,
    roll=True,
    normalize=True,
    wargs=[],
):
    """Short time fourier transform."""
    out = windowed_frames(
        signal,
        n_fft=n_fft,
        frame_size=frame_size,
        hop=hop,
        window=window,
        wargs=wargs,
    )
    if roll:
        out = np.roll(out, -(frame_size // 2), 1)

    # TODO Try sc.fft also
    out = np.apply_along_axis(czt, 1, out, m=n_fft, normalize=normalize)

    return out.T


def stft_tjoa(x, fs, framesz, hop):
    """Short time fourier transform.

    Code from an Stackoverflow answer by Steve Tjoa
    http://stackoverflow.com/questions/2459295/stft-and-istft-in-python?answertab=votes#tab-top
    """
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = np.hamming(framesamp)
    X = sc.array(
        [
            sc.fft(w * x[i:i + framesamp])
            for i in range(0, len(x) - framesamp, hopsamp)
        ]
    )

    return X


def istft_tjoa(X, fs, T, hop):
    """Inverse short time fourier transform.

    Code from an Stackoverflow answer by Steve Tjoa
    http://stackoverflow.com/questions/2459295/stft-and-istft-in-python?answertab=votes#tab-top
    """
    x = sc.zeros(T * fs)
    framesamp = X.shape[1]
    hopsamp = int(hop * fs)
    for n, i in enumerate(range(0, len(x) - framesamp, hopsamp)):
        x[i:i + framesamp] += sc.real(sc.ifft(X[n]))
    return x


def tjoa_demo(signal=None):
    """Demo of the short time fourier transforms."""
    f0 = 440  # Compute the STFT of a 440 Hz sinusoid
    fs = sampler.rate  # sampled at 8 kHz
    # T is 5 seconds by default
    T = int(len(signal) / sampler.rate) if signal is not None else 5
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.020  # and hop size of 20 milliseconds.

    # Create test signal and STFT.
    t = sc.linspace(0, T, T * fs, endpoint=False)

    if signal is None:
        x = sc.sin(2 * sc.pi * f0 * t)  # test signal
    else:
        x = signal

    X = stft_tjoa(x, fs, framesz, hop)

    pylab.interactive(True)

    # Plot the magnitude spectrogram.
    pylab.figure()
    pylab.imshow(
        sc.absolute(np.log(X.T)),
        origin='lower',
        aspect='auto',
        interpolation='nearest',
    )
    pylab.xlabel('Time')
    pylab.ylabel('Frequency')
    pylab.spectral()

    # Compute the ISTFT.
    xhat = istft_tjoa(X, fs, T, hop)

    # Plot the input and output signals over 0.1 seconds.
    T1 = int(0.1 * fs)

    pylab.figure()
    pylab.plot(t[:T1], x[:T1], t[:T1], xhat[:T1])
    pylab.xlabel('Time (seconds)')

    pylab.figure()
    pylab.plot(t[-T1:], x[-T1:], t[-T1:], xhat[-T1:])
    pylab.xlabel('Time (seconds)')
    pylab.show()


def windowed_frames(
    signal, n_fft=2048, frame_size=None, hop=None, window=np.hamming, wargs=[]
):
    """Signal frames windowed with the given window function applied
    using the given frame and hop size.

    Frames are zero padded to n_fft length.
    """
    if frame_size is None:
        # TODO Make n_fft be the next power of two from frame_size
        frame_size = n_fft
    if hop is None:
        hop = frame_size // 2

    assert n_fft > 0, "Number of FFT bins must (n_fft) must be positive."
    assert (
        n_fft >= frame_size
    ), "Frame size must be less than or equal to FFT bin count (n_fft)."
    assert (
        frame_size >= hop
    ), "Hop size must be less than or equal to frame size."

    if isinstance(window, types.FunctionType):
        window_array = window(frame_size, *wargs, sym=False)
    elif is_sequence(window):
        if len(window) == frame_size:
            window_array = np.asarray(window)
        else:
            raise ValueError("Window length must equal frame size.")
    else:
        raise TypeError("Window must be a function or a sequence.")

    frames = sliding_window(signal, frame_size, hop)
    out = window_array * frames

    # Zero pad
    #
    # TODO Make pad function multidimensional by adding axis argument on
    # akasha.math.functions
    pad_size = max(0, n_fft - frame_size)
    out = np.hstack([out, np.zeros((out.shape[0], pad_size))])

    return out
