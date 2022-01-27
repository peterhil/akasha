#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Musical tuning systems module.

See "Pitch Systems in Tonal Music" series on YouTube:
http://www.youtube.com/watch?v=0j-YXgXTpoA
"""

from __future__ import division

import numpy as np
import operator as op

from fractions import Fraction

from akasha.audio.oscillator import Frequency
from akasha.io.keyboard import kb, pos
from akasha.settings import config
from akasha.utils.log import logger
from akasha.utils.python import class_name
from akasha.math import pi2, find_closest_index, map_array


class EqualTemperament():
    """
    Equal temperament tuning:
    http://en.wikipedia.org/wiki/Equal_temperament
    """
    def __init__(self, n=12, scale=2.0):
        self.n = n
        self.__scale = scale

    @property
    def generators(self):
        """
        Get the generators for the lattice.
        """
        return tuple(self.get_generators(self.octave(self.n, self.__scale)))

    @property
    def scale(self):
        """
        Get the interval ratios for one octave.
        """
        return self.octave(self.n, self.__scale)

    @staticmethod
    def get_generators(scale, large=Fraction(3, 2), small=Fraction(9, 8)):
        """
        Find the generators for the lattice from the scale
        closest to the large and the small base interval.
        """
        return scale[map(
            lambda x: find_closest_index(scale, x),
            [large, small]
        )]

    @staticmethod
    def octave(n, scale=2.0):
        """
        Calculate the interval ratios for one octave with n notes of Equal temperament scale.

        The scale parameter defines the biggest interval of the scale.
        It's usually 2.0 for octave, but is 3.0 (a tritave) for example in Bohlen–Pierce scale.
        """
        return scale ** (np.arange(n + 1.0, dtype=np.float64) / n)

    def __repr__(self):
        return f'{class_name(self)}({self.n}, {self.__scale})'

    def __str__(self):
        return f'<{class_name(self)}: {self.n}>'


class LucyTuning():
    """
    Lucy tuning:
    http://www.lucytune.com/

    "The natural scale of music is associated with the ratio of the
    diameter of a circle to its circumference."
    (i.e. pi = 3.14159265358979323846 etc.) -- John Harrison (1693-1776)

    This scale is based on two intervals:

    1) (L), The Larger note as he calls it -- this is the ratio of the 2*pi root of 2,
            which equals a ratio of 1.116633 or 190.9858 cents.

    2) (s), The lesser note, which is half the difference between
            five Larger notes (5L) and an octave
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
        """Large base interval."""
        return 2.0 ** (n / pi2)

    @classmethod
    def s(cls, n):
        """Small base interval."""
        return (2.0 / cls.L(5)) ** (n / 2.0)



class AbstractLayout():
    """
    Abstract base class for musical keyboard layouts.
    """
    def move(self, *pos):
        """
        Move the placement of keys (or origo) on the generator lattice.
        """
        assert len(pos) == 2, "Expected two arguments or tuple of length two."
        self.origo = (self.origo[0] + pos[0], self.origo[1] + pos[1])

    def get_frequency(self, key):
        return self.get(*(pos.get(key, pos[None])))

class WickiLayout(AbstractLayout):
    """
    Wicki-Hayden note layout:
    http://en.wikipedia.org/wiki/Wicki-Hayden_note_layout
    """
    # Why 432 Hz?
    #
    # > factors(432)
    # array([  1,   2,   3,   4,   6,   8,   9,  12,  16,  18,
    #         24,  27,  36, 48,  54,  72, 108, 144, 216, 432])
    #
    # See:
    # http://en.wikipedia.org/wiki/Concert_pitch#Pitch_inflation
    # http://en.wikipedia.org/wiki/Schiller_Institute#Verdi_tuning
    # http://www.mcgee-flutes.com/eng_pitch.html
    #
    # Listen:
    # http://www.youtube.com/results?search_query=432hz&page=&utm_source=opensearch
    # http://www.youtube.com/watch?v=OcDcGsbYA8k
    # http://www.youtube.com/results?search_query=marko+rodin+vortex+math

    # Why 441 or 882?
    #
    # Good for testing with 44100 Hz sampling rate

    def __init__(self, base=Frequency(config.frequency.BASE), origo=(1, 5), generators=(
            # LucyTuning.L(3) * LucyTuning.s(1), LucyTuning.s(1)
            (Fraction(3,2), Fraction(9,8)) # Pythagorean or Just intonation (3-limit)
            # EqualTemperament(5).generators
            # EqualTemperament(12).generators
            # EqualTemperament(19).generators
    )):
        """
        Wicki keyboard layout. Generators are given in (y, x) order.

        Origo defaults to 'C' key, being on the position
        (1,4) + 1 for columns 'tilting' to the left.
        """
        if len(generators) == 2:
            self.base = base
            self.gen = generators
            self.origo = origo
        else:
            raise AttributeError(
                f'Expected two generators, got: {generators!r}'
            )

    def get(self, *pos):
        """
        Get a frequency on key position.
        """
        if pos == kb.shape:
            return Frequency(0.0)
        else:
            return self.base * \
                (self.gen[0] ** (pos[0] - self.origo[0])) * \
                (self.gen[1] ** (pos[1] - self.origo[1]))


class PianoLayout(AbstractLayout):
    """
    Classical piano layout.
    """
    def __init__(self, base=Frequency(config.frequency.BASE), origo=(1, 5)):
        self.base = base
        self.origo = origo
        self.gen = 2 ** (1 / 12.0)

    @property
    def halftones(self):
        return {
            'C': 0, 'C#': 1,
            'D': 2, 'D#': 3,
            'E': 4,
            'F': 5, 'F#': 6,
            'G': 7, 'G#': 8,
            'A': 9, 'A#': 10,
            'B': 11,
            '_': -1,
        }

    @property
    def lattice(self):
        return np.array([
            ['C', 'C#'],
            ['D', 'D#'],
            ['E', '_'],
            ['F', 'F#'],
            ['G', 'G#'],
            ['A', 'A#'],
            ['B', '_'],
        ], dtype='|S2').T

    def get(self, *pos):
        """
        Get a frequency on key position.
        """
        pos = np.subtract(pos, np.array(self.origo))
        key = self.lattice[tuple(np.mod(pos, self.lattice.shape))]
        octave_block = np.floor_divide(pos, self.lattice.shape)
        octave = np.sum(octave_block)

        if key == '_' or tuple(pos) == kb.shape:
            freq = Frequency(0.0)
        else:
            freq = self.base * 2 ** ((octave * 12.0 + self.halftones[key]) / 12.0)

        logger.debug(
            "Playing key '%s%s' (%s) with %s. "
            "Got %s with octave block distance %s.",
            key,
            octave,
            self.halftones[key],
            freq,
            tuple(pos),
            octave_block,
            )

        return freq
