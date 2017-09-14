#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Akasha audio program suite interactive lab.
"""

from __future__ import absolute_import

import numpy as np
import pylab as lab

lab.interactive(True)

import akasha

from cmath import rect, pi, exp, phase
from funckit import xoltar as fx
from scipy.signal import hilbert

from akasha.audio.delay import Delay
from akasha.audio.dtmf import DTMF
from akasha.audio.envelope import Adsr, Exponential, Gamma
from akasha.audio.envelope.beta import Beta, InverseBeta
from akasha.audio.frequency import Frequency
from akasha.audio.harmonics import Harmonics
from akasha.audio.mix import Mix
from akasha.audio.noise import *
from akasha.audio.oscillator import *
from akasha.audio.overtones import Overtones
from akasha.audio.scalar import Scalar
from akasha.audio.resample import Resample
from akasha.audio.sum import Sum

from akasha.control.io.audio import play, write, read
from akasha.control.io.keyboard import *

from akasha.curves import *

from akasha import dsp
from akasha.effects.tape import *
from akasha.effects.filters import *

from akasha.graphic.animation import *
from akasha.graphic.drawing import *
from akasha.graphic.output import *
from akasha.graphic.plotting import *
from akasha.graphic.primitive.line import *
from akasha.graphic.primitive.spline import *

from akasha.math.geometry import *
from akasha.math.geometry.curvature import *

from akasha.net.wiki import *

from akasha.timing import sampler
from akasha.tunings import WickiLayout

from akasha.math import *
