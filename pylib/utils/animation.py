#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
from timeit import default_timer as time
from fractions import Fraction

from funct import pairwise
from graphing import *
from control.io.keyboard import *
from timing import Sampler, time_slice
from tunings import WickiLayout

w = WickiLayout(440.0)
#w = WickiLayout(440.0, generators=(2**(Fraction(7, 12)), 2**(Fraction(2, 12))))

try:
    import pygame
    from pygame import surfarray
    from pygame import sndarray, mixer
    from pygame.locals import *
except ImportError:
    raise ImportError('Error Importing Pygame/surfarray')

def blocksize():
    return int(round(Sampler.rate / float(Sampler.videorate)))

def indices(snd, dur=False):
    if hasattr(snd, "size"):
        size = snd.size
    elif dur:
        size = int(round(dur * Sampler.rate))
    else:
        size = Sampler.rate
    return np.append(np.arange(0, size, blocksize()), size)

def show_slice(screen, snd, size=800, name="Resonance", antialias=True):
    "Show a slice of the signal"
    img = draw(snd, size, antialias=antialias)
    img = img[:,:,:-1]  # Drop alpha

    # screen = pygame.display.set_mode(img.shape[:2], 0, 32)
    surfarray.blit_array(screen, img)
    pygame.display.flip()

def snd_slice(snd, sl):
    return np.cast['int16'](snd[sl].imag * (2**16/2.0-1))

def pcm(snd, bits=16):
    return np.cast['int' + str(bits)](snd.imag * (2**bits/2.0-1))

def anim(snd, size=800, dur=5.0, name="Resonance", antialias=False, lines=False, sync=False):
    sync = (antialias or lines or sync) # For avoiding slowness with colours TODO: optimize colours!
    
    if 'numpy' in surfarray.get_arraytypes():
        surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')
    print ('Using %s' % surfarray.get_arraytype().capitalize())

    pygame.init()
    mixer.quit()
    mixer.init(frequency=Sampler.rate, size=-16, channels=1, buffer=int(round(blocksize()/8.0))) # Keep the buffer smaller than blocksize!
    init = mixer.get_init()
    chs = []
    if hasattr(snd, 'frequency'):
        nchannels = 1
    else:
        nchannels = 1
    for i in xrange(nchannels):
        chs.append(mixer.find_channel())
    chid = 0
    ch = chs[chid]
    
    clock = pygame.time.Clock()
    
    resolution = (size+1, size+1) # FIXME get resolution some other way. This was: img.shape[:2]
    screen = pygame.display.set_mode(resolution) #, flags=pygame.SRCALPHA, depth=32)
    pygame.display.set_caption(name)

    it = pairwise(indices(snd, dur))
    asl = slice(*it.next())
    show_slice(screen, snd[asl], size=size, name=name, antialias=antialias)

    if sync:
        audio = sndarray.make_sound(pcm(snd[asl]))
        ch.play(audio)
    else:
        pgsnd = sndarray.make_sound(pcm(snd[time_slice(dur, 0)]))
        pgsnd.play()

    VIDEOFRAME = pygame.NUMEVENTS - 1
    def set_timer():
        ms = (1.0/Sampler.videorate*1000) # 40 ms for 25 Hz
        pygame.time.set_timer(VIDEOFRAME, int(round(ms))) # FIXME - complain about ints to pygame !!! #1000.0/float(Sampler.videorate))
    set_timer()

    Sampler.paused = False
    def pause():
        Sampler.paused = not Sampler.paused

    done = False
    while not done:
        for event in pygame.event.get():

            # Handle events
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_F8) or \
                 (event.type == pygame.ACTIVEEVENT and event.state == 3):
                # Pause
                pause()
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_F7):
                # Rewind
                it = pairwise(indices(snd))
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_UP):
                # Sampler.videorate += 1
                # set_timer()
                #w.move(-2, 0)
                w.base *= 2.0
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN):
                # Sampler.videorate = max(Sampler.videorate - 1, 1) # Prevent zero division
                # set_timer()
                #w.move(2, 0)
                w.base /= 2.0
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT):
                w.move(0, 6)
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT):
                w.move(0, -6)
            elif event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == 27):
                # Quit
                done = True
                break
            elif (event.type == pygame.KEYDOWN and hasattr(snd, 'frequency')):
                print event
                f = w.get(*( pos.get(event.key, pos[None]) or (0, 0) ))
                snd.frequency = f
                it = pairwise(indices(snd, dur))
                logger.info("Setting NEW frequency: %r for %s, now at frequency: %s" % (f, snd, snd.frequency))
            # elif (event.type == pygame.KEYUP and hasattr(snd, 'frequency')):
            #     print event
            elif (event.type == VIDEOFRAME):
                if Sampler.paused:
                    break
                
                draw_start = time()
                try:
                    asl = slice(*it.next())
                    samples = snd[asl]
                    if sync:
                        audio = sndarray.make_sound(pcm(snd[asl]))
                        # if hasattr(snd, 'frequency'): ch = mixer.find_channel()
                        # chid = (chid + 1) % nchannels
                        # ch = chs[chid]
                        ch.queue(audio)
                except StopIteration:
                    done = True
                    break

                """Do both mechanics and screen update"""
                if lines:
                    img = get_canvas(size, axis=True)[:,:,:-1]  # Drop alpha
                    surfarray.blit_array(screen, img)
                    if antialias: # Colorize
                        for ends in pairwise(samples):
                            ends = np.array(ends)
                            pts = get_points(ends, size).transpose()
                            color = pygame.Color('green') #hsv2rgb(angle2hsv(phase2hues(ends, padding=False)))
                            pygame.draw.aaline(screen, color, *pts)
                    else:
                        pts = get_points(samples, size).transpose()
                        pygame.draw.aalines(screen, pygame.Color('orange'), False, pts, 1)
                    pygame.display.flip()
                else:
                    show_slice(screen, samples, size=size, name=name, antialias=antialias)

                dc = time() - draw_start
                fps = clock.get_fps()
                t = clock.tick_busy_loop(Sampler.videorate)         #cap the framerate
                logger.log(logging.BORING, "Animation: clock tick %d, FPS: %3.3f, drawing took: %.4f", t, fps, dc)
            else:
                print event

    mixer.quit()
    pygame.quit()

if __name__ == '__main__':
    anim()
