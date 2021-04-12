#!/usr/bin/env python
# -*- coding: utf-8 -*-

import faulthandler

from akasha.lab import *
from akasha.utils.planets import *

t = np.arange(0, 10, 1 / sampler.rate)

faulthandler.enable()

# earth = KeplerOrbit(0.14710, 0.14960, 0.0167086, 1)
# anim(normalize(earth.at(t)), lines=True)

a = KeplerOrbit(0.307499, 0.387098, eccentricity=0.0920563, period=0.000240846, name='a')
b = KeplerOrbit(0.407499, 0.987098, eccentricity=0.5640563, period=0.008146, name='b')
c = KeplerOrbit(0.1407499, 0.62798, eccentricity=0.9140563, period=0.005146, name='c')
d = KeplerOrbit(0.0098, 0.1307, eccentricity=0.735, period=0.1, name='d')
e = KeplerOrbit(0.1075, 0.2383, eccentricity=0.4356, period=0.0546, name='e')
f = KeplerOrbit(0.4567, 0.6278, eccentricity=0.6345, period=0.0108, name='f')
orbits = [a, b, c, d, e, f]

oscs = [Osc(1, curve=o) for o in orbits]
solar = Sum(*oscs)

anim(normalize(solar.at(t)), lines=True)
