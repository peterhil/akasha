#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

__package__ = 'akasha'

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


import akasha

from . import settings
from .utils.log import logger

if not hasattr(settings, 'basedir'):
	settings.basedir = os.path.abspath(os.path.dirname(__file__))
	logger.info("Started from: {0}".format(settings.basedir))

from .audio.dtmf import DTMF
from .audio.envelope import Attack, Exponential, Gamma
from .audio.frequency import Frequency
from .audio.harmonics import Overtones
from .audio.noise import *
from .audio.oscillator import *
from .audio.sound import Sound, Group, Pcm

from .control.io.audio import play, write, read
from .control.io.keyboard import *

from .effects.tape import *
from .effects.filters import *

from .funct.xoltar import functional as fx
from .funct.xoltar import lazy

from .graphic.animation import *
from .graphic.drawing import *
from .graphic.primitive.line import *
from .graphic.primitive.spline import *

from .net.wiki import *

from .timing import sampler
from .tunings import WickiLayout

from .utils.math import *

if __name__ == '__main__':
	from akasha import *
