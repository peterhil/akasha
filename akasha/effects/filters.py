#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IIR and other filters.
"""

import numpy as np

from akasha import dsp

from akasha.audio.oscillator import Osc
from akasha.audio.frequency import Frequency, Fraction
from akasha.dsp.z_transform import czt, iczt
from akasha.timing import sampler
from akasha.utils.log import logger
from akasha.utils.math import pi2, get_impulses, normalize, complex_as_reals, as_complex, pad


def unosc(signal):
    """
    Return unwrapped log signal, that can be fed
    back to complex exponentiation after transformations.
    """
    s = np.log(signal)

    #d = distances(s / pi2)
    #impulses = np.round(d - d[0])
    #impulses = np.fmax(np.sign(d-0.9999), 0)
    #impulses = np.fmax(np.sign(d - 0.5), 0)

    #impulses = distances(np.sign((distances(np.angle(s)) - np.angle(s[1:]))))/2
    #impulses = (np.angle(signal) - (np.angle(signal) % np.pi)) / np.pi * -3 - 1

    #unwrap = (-np.cumsum(pad(impulses, 0)))[:-2]
    #unwrap = np.cumsum(distances((np.sign((distances(np.angle(s)) - np.angle(s[1:])))+1)/2))

    impulses = get_impulses(signal, tau=False)
    unwrap = np.cumsum(impulses)

    return s.real + ((s.imag % pi2) + unwrap) * 1j


def log_plane(freq, amp1=1, amp2=1):
    """
    Return the complex logarithm of a signal from given frequencies (at amplitude 1).
    Result can be fed to back to np.exp to get the sum of oscillators (additive synthesis of frequencies).

    To test:
    >>> o882 = Osc(882)
    >>> o1764 = Osc(1764)
    >>> s = (o882[:50] + o1764[:50]) / 2
    >>> unosc(s) - log_plane(882)
    # Should be almost all zero
    """
    # TODO Get (multiple) frequencies from arguments
    # i1 = Frequency.angles(Fraction(freq, sampler.rate)) * pi2
    # i2 = Frequency.angles(Fraction(freq * 2, sampler.rate)) * pi2
    # TODO Filter zero frequencies
    def ifrequency(freq):
        return np.arange(0, 1, float(freq) / sampler.rate) * pi2 * 1j
    if freq != 0:
        i1 = np.log(amp1) + ifrequency(freq)
        i2 = np.log(amp2) + ifrequency(freq * 2)
    else:
        i1 = i2 = np.zeros(1, dtype=np.float64)
    i2 = np.hstack([i2, i2])

    angle_diff = np.abs(i2.imag - i1.imag)
    real = np.log(np.sin((angle_diff + np.pi) / 2))
    imag = (i1 + i2) / 2
    return real + imag


def freq_shift(signal, a=12.1, b=0.290147):
    """
    Shift frequencies of a signal without affecting time.
    """
    #f = 440
    #scale = (1+f/sampler.rate)
    w = unosc(signal)
    # remove - before exp and - np.pi?
    out = -np.exp(w.real + (normalize((w.imag * a) % (pi2 / b)) * pi2 - np.pi) * 1j)

    return out


def transform(signal, scale=np.array([1, 1]), translate=np.array([0, 0])):
    """
    Transform a signal with scale and/or translate in the frequency domain (complex plane).
    """
    # E1103: Instance of 'list' has no 'T' member (but some types could not be inferred)
    # pylint: disable=E1103

    # TODO: Affine transformations! http://en.wikipedia.org/wiki/Affine_transformation

    tr = complex_as_reals(unosc(signal))
    tr *= np.atleast_2d(scale).T
    tr += np.atleast_2d(translate).T

    return np.exp(as_complex(tr))


def highpass(signal, freq, bins=256, pass_zero=True, scale=False, nyq=sampler.rate / 2.0):
    """
    Highpass filter.
    """
    a = 1
    b = dsp.signal.firwin(bins, cutoff=freq, pass_zero=pass_zero, scale=scale, nyq=nyq)
    return dsp.signal.lfilter(b, a, signal)


def lowpass(signal, cutoff=sampler.rate / 2.0, bins=256):
    """
    Lowpass filter.
    """
    fs = float(sampler.rate)
    fc = cutoff / fs
    a = 1
    b = dsp.signal.firwin(bins, cutoff=fc, window='hamming')
    return dsp.signal.lfilter(b, a, signal)


def resonate(signal, poles, zeros=np.array([]), gain=1.0, axis=-1, zi=None):
    """
    Resonate a signal with an IIR filter based on given poles and zeros.

    Usage example
    =============

    bjork = read('Bjork - Human Behaviour.aiff', dur=30)
    poles = pole_frequency(np.array([30, 50, 238., 440., 1441]), [0.99, 0.999, 0.99, 0.87, 0.77])
    anim(normalize(resonate(bjork, poles, gain=1.0)))
    """
    b, a = dsp.signal.filter_design.zpk2tf(zeros, poles, gain)
    logger.debug("Resonate: order: {},\n\tb: {},\n\ta: {}".format(max(len(a), len(b)), b, a))
    if zi == 'auto':
        zi = dsp.signal.lfilter_zi(b, a)
        return dsp.signal.lfilter(b, a, signal, axis=axis, zi=zi)[0]
    else:
        return dsp.signal.lfilter(b, a, signal)


def resonator_comb(
        signal,
        a=5,
        b=6,
        step=135,
        dampen=1 - 1 / 2 ** 15,
        sp_dur=2,
        fx_dur=4,
        playtime=10):
    """
    Apply a resonating IIR filter comb to a signal.
    """
    # Is dampen double the Q value?
    roots = Osc(1).sample
    fs = sampler.rate
    out = normalize(
        pad(signal[:int(round(sp_dur * fs))], -1, int(round(playtime * fs - sp_dur * fs)), 0) +
        normalize(
            resonate(
                pad(signal[:fx_dur * fs], -1, playtime * fs - fx_dur * fs, 0),
                roots[a:a + b * step:step] * dampen,
                [-1, 1j]
            )
        )
    )
    #anim(out, dur=playtime, antialias=False)

    return out[:int(round(playtime * fs))]


def resonator_p1(signal, pole, m=None, gain=1.0):
    """
    Complex one pole resonator
    https://ccrma.stanford.edu/~jos/filters/Complex_Resonator.html

    Note: To get the results normalized to unit circle, set gain to:
    gain = 1.0/(1 - np.abs(pole))

    Examples
    ========

    * Windy noise:
    n = Noise()
    p = pole_frequency(59)
    g = 0.99994
    r = resonator_p1(f, p*g, m=30*sampler.rate, gain=rect(1, 1.0/300*pi2))
    anim normalize(r)

    * Hoover or car noise:
    rn = Rustle(0.545, 275, Exponential(-0.05))
    f  = rn[:10*44100]
    g = 0.99894
    r = resonator_p1(f, p*g, m=10*sampler.rate, gain=rect(1, 1.0/300*pi2))
    anim normalize(r)
    """
    signal = np.atleast_1d(signal).astype(np.complex)
    if m is None: m = len(signal)
    n = np.arange(m)

    response = dsp.unit_step(n) * gain * pole ** n
    return iczt(czt(signal, m=m) * czt(response, m=m))
