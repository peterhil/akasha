#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal as dsp

from audio.oscillator import Osc
from timing import Sampler
from utils.math import *


def unosc(signal):
    """Return unwrapped log signal, that can be fed back to complex exponentiation after transformations."""
    s = np.log(signal)

    #d = distances(s / PI2)
    #impulses = np.round(d - d[0])
    #impulses = np.fmax(np.sign(d-0.9999), 0)
    #impulses = np.fmax(np.sign(d - 0.5), 0)

    #impulses = distances(np.sign((distances(np.angle(s)) - np.angle(s[1:]))))/2
    #impulses = (np.angle(signal) - (np.angle(signal) % np.pi)) / np.pi * -3 - 1

    #unwrap = (-np.cumsum(pad(impulses, 0)))[:-2]
    #unwrap = np.cumsum(distances((np.sign((distances(np.angle(s)) - np.angle(s[1:])))+1)/2))

    impulses = get_impulses(signal, tau=False)
    unwrap = np.cumsum(impulses)
    return s.real + ((s.imag % PI2) + unwrap) * 1j

def freq_shift(signal, a=12.1, b=0.290147):
    #f = 440
    #scale = (1+f/Sampler.rate)
    w = unosc(signal)
    out = -np.exp(w.real + (normalize((w.imag * a) % (PI2/b)) * PI2 - np.pi) * 1j) # remove - before exp and - np.pi?
    return out

def transform(signal, scale=[1, 1], translate=[0, 0]):
    # TODO affine transformations!
    # http://en.wikipedia.org/wiki/Affine_transformation
    signal = signal.copy()
    cx_plane = unosc(signal)
    tr = complex_as_reals(cx_plane)
    tr *= np.atleast_2d(scale).T #np.array([[1, 0.5]]).T
    tr += np.atleast_2d(translate).T
    return np.exp(cx_plane)

def highpass(signal, freq, bins=256, pass_zero=True, scale=False, nyq=Sampler.rate/2.0):
    a = 1
    b = dsp.firwin(bins, cutoff=freq, pass_zero=pass_zero, scale=scale, nyq=nyq)
    return dsp.lfilter(b, a, signal)

def lowpass(signal, cutoff=Sampler.rate/2.0, bins=256):
    fs = float(Sampler.rate)
    fc = cutoff / fs
    a = 1
    b = dsp.firwin(bins, cutoff=fc, window='hamming')
    y = dsp.lfilter(b, a, signal)
    return y

def resonate(signal, poles, zeros=[], gain=1.0, zi=None):
    b, a = dsp.filter_design.zpk2tf(zeros, poles, gain)
    print b, a, max(len(a), len(b))
    #zi = dsp.lfilter_zi(b, a)
    if zi == None:
        return dsp.lfilter(b, a, signal, axis=0, zi=zi)
    else:
        return dsp.lfilter(b, a, signal)

def resonator_comb(signal, a=5, b=6, step=135, dampen=1-1/2**15, sp_dur = 2, fx_dur=4, playtime = 10):
    # Is dampen double the Q value?
    roots = Osc(1).sample
    fs = Sampler.rate
    out = normalize(
        pad(signal[:int(round(sp_dur*fs))], -1, int(round(playtime*fs - sp_dur*fs)), 0) +
        normalize(
            resonate(
                pad(signal[:fx_dur*fs], -1, playtime*fs-fx_dur*fs, 0),
                roots[a:a+b*step:step] * dampen,
                [-1, 1j]
            )
        )
    )
    #anim(out, dur=playtime, antialias=False)
    return out[:int(round(playtime*fs))]

