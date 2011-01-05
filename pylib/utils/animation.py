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

def anim(snd = None):
    if (snd == None): snd = self.make_test_sound()
    
    if 'numpy' in surfarray.get_arraytypes(): surfarray.use_arraytype('numpy')
    else: raise ImportError('Numpy array package is not installed')

    pygame.init()
    print ('Using %s' % surfarray.get_arraytype().capitalize())
    
    def surfdemo_show(array_img, name):
        "displays a surface, waits for user to continue"
        array_img = array_img[:,:,:-1]  # Drop alpha
        
        screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
        surfarray.blit_array(screen, array_img)
        pygame.display.flip()
        pygame.display.set_caption(name)

        # clock = pygame.time.Clock()        
        pygame.time.set_timer(pygame.USEREVENT+1, int(1.0/Sampler.videorate))
        while 1:
            #check for quit'n events
            # event = pygame.event.wait()
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    return 'quit'
                    # pygame.quit()
                elif event.type == pygame.USEREVENT+1:
                    """Do both mechanics and screen update"""
                    return
                elif event.type == MOUSEBUTTONDOWN:
                    return

                # elif event.type == KEYDOWN and event.key == K_s:
                #     pass
                #     main_dir = os.path.split(os.path.abspath(__file__))[0]
                #     pygame.image.save(screen, main_dir + name + '.png')
        
                #cap the framerate
                # clock.tick(1.0/Sampler.videorate)

    #allblack
    # allblack = np.zeros((128, 128), np.int32)
    # surfdemo_show(allblack, 'allblack')
    
    print snd

    if hasattr(snd, "size"):
        size = snd.size
    else:
        size = 44100
    
    for i in xrange(0, size, int(blocksize())):
        end = min(i + blocksize() * 4, size) - 1
        sl = slice(i, end)
        print sl

        img = draw(normalize(snd[sl]), size=1200)
        res = surfdemo_show(img, 'harmonics')
        if res == 'quit':
            break
    
    #alldone
    pygame.quit()

if __name__ == '__main__':
    anim()
