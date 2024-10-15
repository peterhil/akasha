#!/usr/bin/env python

"""
Dual-tone Multifrequency Tones
"""

from akasha.audio.generators import Generator
from akasha.timing import sampler
from akasha.utils.python import _super


class DTMF(Generator):
    """Dual-tone Multifrequency Tones

    DTMF keypad frequencies:

    Hz   1209  1336  1477  1633
    697     1     2     3     A
    770     4     5     6     B
    852     7     8     9     C
    941     *     0     #     D

    - - -
    Special tones:

    Event        Low frequency  High frequency
    Busy signal         480 Hz  620 Hz
    Ringback tone (US)  440 Hz  480 Hz
    Dial tone           350 Hz  440 Hz

    - - -
    Special Information Tones (AT&T/Bellcore SIT)

    Segment durations:
    Short duration = 276 ms
    Long duration = 380 ms

    Frequencies for use in SITs:
    First segment       Second segment      Third segment
    (high) 985.2 Hz     (high) 1428.5 Hz
    (low) 913.8 Hz      (low) 1370.6 Hz     (low) 1776.7 Hz

    The interval between the segments of SITs is between 0 and 4 ms.
    To minimize the number of callers who may abandon the call without
    listening to the announcement, the nominal time gap between the
    third tone segment and the beginning of the announcement is set as
    close to zero as possible, with an allowed maximum of 100 ms.

    - - -
    A standard telephone keypad

    Most of the keys also bear letters according to the following system:
    0 = none (in some telephones, "OPERATOR" or "OPER") (gsm: space)
    1 = none (in some older telephones, QZ)
    2 = ABC
    3 = DEF
    4 = GHI
    5 = JKL
    6 = MNO
    7 = P(Q)RS
    8 = TUV
    9 = WXY(Z)
    * = (gsm: +)
    # = (gsm: shift)
    """

    sp = [350, 440, 480, 620]
    lo = [697, 770, 852, 941]
    hi = [1209, 1336, 1477, 1633]

    nkeys = '123A' + '456B' + '789C' + '*0#D'

    table = {}

    for i, key in enumerate(nkeys):
        l, h = divmod(i, 4)
        table[key] = (lo[l], hi[h])

    keys = b''.join(
        [
            b'ABC',
            b'DEF',
            b'GHI',
            b'JKL',
            b'MNO',
            b'PQRS',
            b'TUV',
            b'WXYZ',
            b' ',
            b'-',
        ]
    )
    digits = b''.join(
        [
            b'222',
            b'333',
            b'444',
            b'555',
            b'666',
            b'7777',
            b'888',
            b'9999',
            b'0',
            b'-',
        ]
    )
    alphabet_trans = bytes.maketrans(keys, digits)

    def __init__(self, sequence, pulselength=0.07, pause=0.05):
        _super(self).__init__()

        self.sequence = sequence.upper()
        self.pulselength = pulselength
        self.pause = pause

    @property
    def number(self):
        """The number to dial."""
        return self.sequence.upper().translate(self.alphabet_trans)

    def sample(self, frames):
        """Make DTMF dialing tone."""

    def __len__(self):
        pulselengths = len(self.number) * (self.pulselength + self.pause)
        length = pulselengths * sampler.rate - self.pause
        return int(round(length))
