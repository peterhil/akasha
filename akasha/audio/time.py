#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from akasha.types.numeric import AlgebraicField
from cdecimal import Decimal, getcontext


getcontext().prec = 32


class Chrono(AlgebraicField):
    """
    Chrono is a nanosecond precision time class, that is compatible with OSC.

    Internal format is 64-bit fixed number with 32 bits for seconds and
    another 32 bits for parts of a second.
    """
    def __init__(self, time):
        self._unit = 'time'
        self.time = Decimal(time)
