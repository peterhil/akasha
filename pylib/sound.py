#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from fractions import Fraction

# My modules
from envelope import Exponential
from oscillator import Osc
from harmonics import Harmonic
from generators import Generator
from utils import play, write

# np.set_printoptions(precision=4, suppress=True)

class Sound(object, Generator):
    """Sound groups."""
    
    def __init__ (self, *args):
        self.sounds = list(args)
        
    def sample(self, iter):
        '''Pass parameters to all sound objects and update states.'''
        sound = np.zeros(len(iter), dtype=complex)
        for sndobj in self.sounds:
            sound += sndobj[iter]
        return sound / max( len(self.sounds), 1.0 )
        
    # def __repr__(self):
    #     components = u''
    #     for obj in self.sounds:
    #         components += repr(obj)
    #     return "Sound(%s)" % (components)
    
    # Use s.sounds.append(sndobj)
    # s.sounds.index(sndobj)
    
    # def add(self, sndobj, offset=0):
    #     '''Add a new sndobj to self.'''
    #     self.sounds[sndobj] = sndobj
    # 
    # def stat(self):
    #     '''Return a tuple containing the state of each sound object.'''
    #     return tuple(self.sounds.values())