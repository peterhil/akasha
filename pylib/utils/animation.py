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

def blocksize():
    return int(round(Sampler.rate / float(Sampler.videorate)))

def anim(snd = None):
    def make_test_sound():
        freq=230
        h = Harmonic(freq, damping=lambda f, a=1.0: (-f/100.0, a/(f/freq)), n = 20)(230)
        c = Chaos()
        o = Osc.freq(220)
        s = Sound(h, o)
        return s
    
    snd = snd or make_test_sound()
    
    main_dir = os.path.split(os.path.abspath(__file__))[0]

    if 'numpy' in surfarray.get_arraytypes():
        surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')

    pygame.init()
    print ('Using %s' % surfarray.get_arraytype().capitalize())
    
    def surfdemo_show(array_img, name):
        "displays a surface, waits for user to continue"
        array_img = array_img[:,:,:-1]  # Drop alpha
        
        screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
        surfarray.blit_array(screen, array_img)
        pygame.display.flip()
        pygame.display.set_caption(name)
        while 1:
            e = pygame.event.wait()
            if e.type == MOUSEBUTTONDOWN: break
            elif e.type == KEYDOWN and e.key == K_s:
                pygame.image.save(screen, name+'.png')
            elif e.type == QUIT:
                raise SystemExit()

    #allblack
    # allblack = np.zeros((128, 128), np.int32)
    # surfdemo_show(allblack, 'allblack')
    
    print snd
    size = snd[:44100].size
    
    for i in xrange(0, size, int(blocksize())):
        print i
        img = draw(normalize(snd[ i : max(i + blocksize() * 4.0, size) - 1]), size=1200)
        surfdemo_show(img, 'harmonics')
    
    #alldone
    pygame.quit()

if __name__ == '__main__':
    anim()
