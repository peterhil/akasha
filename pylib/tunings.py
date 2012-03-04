#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction
from exceptions import AttributeError

from audio.oscillator import Frequency
from control.io.keyboard import kb
from utils.log import logger
from utils.math import PI2, find_closest_index

# See "Pitch Systems in Tonal Music" series on YouTube:
# http://www.youtube.com/watch?v=0j-YXgXTpoA&feature=related

def cents(*args):
    """Calculate cents from interval or frequency ratio(s).
    When using frequencies, give greater frequencies first.

    >>> cents(float(Fraction(5,4)))
    array([ 386.31371386])

    >>> cents(440/27.5)
    array([ 4800.])     # equals four octaves

    >>> cents(np.arange(8)/16.0+1)
    array([[   0.        ,  104.9554095 ,  203.91000173,  297.51301613,
             386.31371386,  470.78090733,  551.31794236,  628.27434727]])
    """
    return 1200 * np.log2(args)

def interval(*cnt):
    """Calculate interval ratio from cents.

    >>> interval(100)
    array([ 1.05946309])    # one equal temperament semitone

    >>> interval(386.31371386)
    array([ 1.25])          # 5:4, perfect fifth

    >> [map(Fraction.limit_denominator, map(Fraction.from_float, i)) for i in interval(np.arange(5) * 386.31371386)]
    [[Fraction(1, 1),
      Fraction(5, 4),
      Fraction(25, 16),
      Fraction(125, 64),
      Fraction(625, 256)]]
    """
    return np.power(2, np.asanyarray(cnt)/1200.0)

def freq_plus_cents(f, cnt):
    """Calculate freq1 + cents = freq2"""
    return f * interval(cnt)

class EqualTemperament(object):
    def __init__(self, n = 12, scale = 2.0):
        self.n = n
        self.__scale = scale

    @property
    def generators(self):
        return tuple(self.get_generators(self.octave(self.n, self.__scale), self.n))

    @property
    def scale(self):
        return self.octave(self.n, self.__scale)

    @staticmethod
    def get_generators(scale, n, large=Fraction(3, 2), small=Fraction(9, 8)):
        return scale[map(lambda x: find_closest_index(scale, x), [large, small] )]

    @staticmethod
    def octave(n, scale = 2.0):
        return scale ** (np.arange(n + 1.0, dtype=np.float64) / n)

# In [128]: sorted(Fraction(3,2) ** np.array(map(Fraction.from_float, xrange(28))) % Fraction(3,2) + Fraction(1))
# Out[128]:
# [1.0,
#  1.09375,
#  1.12890625,
#  1.1682940423488617,
#  1.2524410635232925,
#  1.375,
#  1.3850951194763184,
#  1.3918800354003906,
#  1.5625,
#  1.5859375,
#  1.6650390625,
#  1.746337890625,
#  1.75,
#  1.890625,
#  1.92926025390625,
#  1.943359375,
#  1.99755859375,
#  2.0,
# ]

class LucyTuning(object):
    """
    http://www.lucytune.com/

    "The natural scale of music is associated with the ratio of the diameter of a circle to its circumference."
    (i.e. pi = 3.14159265358979323846 etc.) -- John Harrison (1693-1776)
    
    This scale is based on two intervals:
    
    1) (L), The Larger note as he calls it -- this is the ratio of the 2*pi root of 2,
       which equals a ratio of 1.116633 or 190.9858 cents.

    2) (s), The lesser note, which is half the difference between five Larger notes (5L) and an octave.
       giving a ratio of 1.073344 or 122.5354 cents.

    The fifth (V) is composed of three Large (3L) plus one small note (s) i.e. (3L+s)
        = (190.986*3) + (122.535)
        = 695.493 cents or ratio of 1.494412.
    The fourth (IV) is 2L+s
        = 504.507 cents.

    Frequencies and ratios:
    http://www.lucytune.com/midi_and_keyboard/frequency_ratios.html
    http://www.lucytune.com/new_to_lt/pitch_04.html
    """
    @classmethod
    def L(cls, n):
        return 2.0 ** (n / PI2)

    @classmethod
    def s(cls, n):
        return (2.0 / cls.L(5)) ** (n / 2.0)

class WickiLayout(object):
    def __init__(self, base=Frequency(432.0), origo=(1, 5), generators=(Fraction(3,2), Fraction(9,8))):
        """Wicki keyboard layout. Generators are given in (y, x) order.
        Origo defaults to 'C' key, being on the position(1,4 + 1 for columns 'tilting' to the left)."""
        if len(generators) == 2:
            self.base = base
            self.gen = generators
            self.origo = origo
        else:
            raise AttributeError("Expected two generators, got: {0!r}".format(generators))
    
    def get(self, *pos):
        logger.debug("Getting position: %s %s" % pos)
        if pos == kb.shape:
            return Frequency(0.0)
        else:
            return (
                self.base * \
                    (self.gen[0] ** (pos[0] - self.origo[0])) * \
                    (self.gen[1] ** (pos[1] - self.origo[1]))
            )

    def move(self, *pos):
        assert len(pos) == 2, "Expected two arguments or tuple of length two."
        self.origo = (self.origo[0] + pos[0], self.origo[1] + pos[1])
