#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast fourier transforms.
"""

import numpy as np
import scipy as sc
import pylab

from akasha.dsp.window import sliding_window
from akasha.dsp.z_transform import czt
from akasha.timing import sampler


def stft(signal, n_fft=2048, frame_size=None, hop=None, window=sc.signal.hamming, sym=False, roll=True, normalize=True, wargs=[]):
    """
    Short time fourier transform.
    """
    if frame_size is None: frame_size = n_fft
    if hop is None: hop = frame_size // 2
    assert n_fft >= frame_size, "Frame size must be less than or equal to FFT bin count (n_fft)."
    assert frame_size >= hop, "Hop size must be less than or equal to frame size."

    frames = sliding_window(signal, frame_size, hop)
    window_array = window(frame_size, *wargs, sym=sym)  # TODO Enable using arrays
    out = window_array * frames
    pad_size = max(0, n_fft - frame_size)
    out = np.hstack([out, np.zeros((out.shape[0], pad_size))])  # Zero pad
    if roll:
        out = np.roll(out, -(frame_size // 2), 1)
    out = np.apply_along_axis(czt, 1, out, m=n_fft, normalize=normalize)  # TODO Try sc.fft also
    return out[::, :int(round(n_fft / 2.0))].T


def stft_tjoa(x, fs, framesz, hop):
    """
    Short time fourier transform.

    Code from an Stackoverflow answer by Steve Tjoa
    http://stackoverflow.com/questions/2459295/stft-and-istft-in-python?answertab=votes#tab-top
    """
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = sc.hamming(framesamp)
    X = sc.array(
        [sc.fft(w * x[i:i + framesamp]) for i in range(0, len(x) - framesamp, hopsamp)]
    )
    return X


def istft_tjoa(X, fs, T, hop):
    """
    Inverse short time fourier transform.

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
    """
    Demo of the short time fourier transforms.
    """
    f0 = 440  # Compute the STFT of a 440 Hz sinusoid
    fs = sampler.rate  # sampled at 8 kHz
    T = int(len(signal)/sampler.rate) if signal is not None else 5  # lasting 5 seconds
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
    pylab.imshow(sc.absolute(np.log(X.T)), origin='lower', aspect='auto',
                 interpolation='nearest')
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
