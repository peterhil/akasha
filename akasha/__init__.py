#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import locale
import logging
import numpy as np
import types
import os
import sys

from cmath import rect, pi, exp, phase
from collections import defaultdict
from fractions import Fraction
from numbers import Number
from scipy.signal import hilbert


from akasha import settings
from akasha.utils.log import logger

if not hasattr(settings, 'basedir'):
    settings.basedir = os.path.abspath(os.path.dirname(__file__))
    logger.info("Started from: {0}".format(settings.basedir))

settings.setup()

from akasha.audio.curves import *
from akasha.audio.dtmf import DTMF
from akasha.audio.envelope import Attack, Exponential, Gamma
from akasha.audio.frequency import Frequency
from akasha.audio.harmonics import Overtones
from akasha.audio.noise import *
from akasha.audio.oscillator import *
from akasha.audio.sound import Sound, Group, Pcm

from akasha.control.io.audio import play, write, read
from akasha.control.io.keyboard import *

from akasha.effects.tape import *
from akasha.effects.filters import *

from akasha.funct.xoltar import functional as fx
from akasha.funct.xoltar import lazy

from akasha.graphic.animation import *
from akasha.graphic.drawing import *
from akasha.graphic.primitive.line import *
from akasha.graphic.primitive.spline import *

from akasha.net.wiki import *

from akasha.timing import sampler
from akasha.tunings import WickiLayout

from akasha.utils.math import *


if __name__ == '__main__':
    pass
