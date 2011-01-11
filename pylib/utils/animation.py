#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os

try:
    import pygame
    from pygame import surfarray
    from pygame.locals import *
except ImportError:
    raise ImportError('Error Importing Pygame/surfarray')

from timing import Sampler

from harmonics import Harmonic
from sound import Sound
from noise import Chaos
from oscillator import Osc

from graphing import *
# from itertools import *
from utils import pairwise

def make_test_sound():
    freq=230
    h = Harmonic(freq, damping=lambda f, a=1.0: (-f/100.0, a/(f/freq)), n = 20)(230)
    c = Chaos()
    o2 = Osc.freq(220)
    o4 = Osc.freq(440)
    o3 = Osc.freq(330)
    s = Sound(h, o2, o3, o4)
    return s

def blocksize():
    return int(round(Sampler.rate / float(Sampler.videorate)))

def indices(snd):
    if hasattr(snd, "size"):
        size = snd.size
    else:
        size = 44100
    return np.append(np.arange(0, size, blocksize()), size)

def show(snd, size=1200, name="Resonance"):
    "Show a slice of the signal"
    img = draw(snd, size, antialias=True)
    img = img[:,:,:-1]  # Drop alpha
    
    screen = pygame.display.set_mode(img.shape[:2], 0, 32)
    surf = surfarray.blit_array(screen, img)
    pygame.display.flip()
    pygame.display.set_caption(name)
    del img, screen, surf
    return False

def anim(snd = None, size="600", name="Resonance"):
    if (snd == None): snd = self.make_test_sound()
    
    if 'numpy' in surfarray.get_arraytypes():
        surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')
    pygame.init()
    print ('Using %s' % surfarray.get_arraytype().capitalize())
    
    it = pairwise(indices(snd))
    # clock = pygame.time.Clock()
    show(snd[slice(*it.next())], size=size, name=name)
    
    pygame.time.set_timer(pygame.USEREVENT, int(round(1.0/Sampler.videorate*1000)))
    while 1:
        event = pygame.event.wait()
        #check for quit'n events
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # pygame.quit()
            break
        elif event.type in [pygame.USEREVENT, MOUSEBUTTONDOWN]:
            """Do both mechanics and screen update"""
            try:
                show(snd[slice(*it.next())], size=size, name=name)
            except StopIteration:
                pygame.time.delay(2000)
                break

        #cap the framerate
        # clock.tick(int(1.0/Sampler.videorate*1000))

    print snd
    
    #alldone
    pygame.quit()

if __name__ == '__main__':
    anim()
