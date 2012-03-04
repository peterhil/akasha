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
from utils.math import pcm

# Why 432 Hz?
#
# > factors(432)
# array([  1,   2,   3,   4,   6,   8,   9,  12,  16,  18,  24,  27,  36, 48,  54,  72, 108, 144, 216, 432])
#
# See:
# http://en.wikipedia.org/wiki/Concert_pitch#Pitch_inflation
# http://en.wikipedia.org/wiki/Schiller_Institute#Verdi_tuning
# http://www.mcgee-flutes.com/eng_pitch.html
#
# Listen:
# http://www.youtube.com/results?search_query=432hz&page=&utm_source=opensearch
# http://www.youtube.com/watch?v=OcDcGsbYA8k
# http://www.youtube.com/results?search_query=marko+rodin+vortex+math

w = WickiLayout(432.0)

# Why 441 or 882? Good for testing with 44100 Hz sampling rate
#w = WickiLayout(441.0, generators=(2**(Fraction(7, 12)), 2**(Fraction(2, 12))))

try:
    import pygame
    from pygame import surfarray
    from pygame import sndarray, mixer
    from pygame.locals import *
    pygame.mixer.pre_init(44100, -16, 1, 128) # Set mixer defaults
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

def show_slice(screen, snd, size=800, name="Resonance", antialias=True, lines=False):
    "Show a slice of the signal"
    #img = draw(snd, size, antialias=antialias, lines=lines)

    # img = get_canvas(size)
    # img = img[:,:,:-1]  # Drop alpha
    # surfarray.blit_array(screen, img)
    img = draw(snd, size, antialias=antialias, lines=lines, screen=screen)

    # screen = pygame.display.set_mode(img.shape[:2], 0, 32)
    img = img[:,:,:-1]  # Drop alpha
    surfarray.blit_array(screen, img)
    pygame.display.flip()

def show_transfer(screen, snd, size=720, name="Transfer", type='PAL', axis='imag'):
    "Show a slice of the signal"
    img = get_canvas(size)
    tfer = video_transfer(snd, type=type, axis=axis)
    black = (size - tfer.shape[0]) / 2.0
    img[1+black:-black,1:,:] = tfer

    img = img[:,:,:-1]  # Drop alpha
    # screen = pygame.display.set_mode(img.shape[:2], 0, 32)
    surfarray.blit_array(screen, img)
    pygame.display.flip()

def init_mixer(*args):
    pygame.init()
    mixer.quit()
    print args[0]
    if args[0]:
        mixer.init(*args[0])
    else:
        mixer.init(frequency=Sampler.rate, size=-16, channels=1, buffer=128) #int(round(blocksize()/8.0))) # Keep the buffer smaller than blocksize!
    init = mixer.get_init()
    logger.debug("Mixer init: %s Sampler rate: %s Video rate: %s" % (init, Sampler.rate, Sampler.videorate))
    return init

def anim(snd, size=800, dur=5.0, name="Resonance", antialias=True, lines=False, sync=True, init=None):
    """
    Animate complex sound signal
    """
    if 'numpy' in surfarray.get_arraytypes():
        surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')
    logger.info("Using %s" % surfarray.get_arraytype().capitalize())

    init_mixer(init)
    pygame.display.set_caption(name)

    clock = pygame.time.Clock()
    resolution = (size+1, size+1) # FIXME get resolution some other way. This was: img.shape[:2]
    screen = pygame.display.set_mode(resolution) #, flags=pygame.SRCALPHA, depth=32)

    chs = []
    if hasattr(snd, 'frequency'):
        nchannels = 1
    else:
        nchannels = 1
    for i in xrange(nchannels):
        chs.append(mixer.find_channel())
    chid = 0
    ch = chs[chid]
    
    it = pairwise(indices(snd, dur))
    asl = slice(*it.next())
    show_slice(screen, snd[asl], size=size, name=name, antialias=antialias, lines=lines)

    if sync:
        audio = sndarray.make_sound(pcm(snd[asl]))
        ch.play(audio)
    else:
        pgsnd = sndarray.make_sound(pcm(snd[time_slice(dur, 0)]))
        pgsnd.play()

    VIDEOFRAME = pygame.NUMEVENTS - 1
    def set_timer():
        # FIXME - complain about ints to pygame
        ms = int(round(1000.0 / Sampler.videorate)) # 40 ms for 25 Hz
        pygame.time.set_timer(VIDEOFRAME, ms)
    set_timer()

    Sampler.paused = False
    def pause():
        Sampler.paused = not Sampler.paused
        logger.info("Pause" if Sampler.paused else "Play")

    def change_frequency(snd, key):
        f = w.get(*( pos.get(key, pos[None]) or (0, 0) ))
        snd.frequency = f
        logger.info("Setting NEW frequency: %r for %s, now at frequency: %s" % (f, snd, snd.frequency))

    done = False
    while not done:
        for event in pygame.event.get():

            # Quit
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == 27):
                done = True
                break

            # Pause
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_F8) or \
                 (event.type == pygame.ACTIVEEVENT and event.state == 3):
                pause()

            # -- Key downs --
            elif event.type == pygame.KEYDOWN:
                logger.debug("Key down: %s" % event)
                # Rewind
                if pygame.K_F7 == event.key:
                    it = pairwise(indices(snd))
                    logger.info("Rewind")
                # Arrows
                elif pygame.K_UP == event.key:
                    # Sampler.videorate += 1
                    # set_timer()
                    #w.move(-2, 0)
                    w.base *= 2.0
                elif pygame.K_DOWN == event.key:
                    # Sampler.videorate = max(Sampler.videorate - 1, 1) # Prevent zero division
                    # set_timer()
                    #w.move(2, 0)
                    w.base /= 2.0
                elif pygame.K_LEFT == event.key:
                    w.move(0, 1)
                elif pygame.K_RIGHT == event.key:
                    w.move(0, -1)
                # Change frequency
                elif hasattr(snd, 'frequency'):
                    change_frequency(snd, event.key)
                    it = pairwise(indices(snd, dur))

            # -- Key ups --
            elif (event.type == pygame.KEYUP and hasattr(snd, 'frequency')):
                if pygame.K_CAPSLOCK == event.key:
                    change_frequency(snd, event.key)
                    it = pairwise(indices(snd, dur))
                else:
                    logger.debug("Key up:   %s" % event)

            # Video frame
            elif (event.type == VIDEOFRAME):
                if Sampler.paused:
                    break
                
                draw_start = time()
                try:
                    asl = slice(*it.next())
                    samples = snd[asl]
                    if sync:
                        audio = sndarray.make_sound(pcm(snd[asl]))
                        if hasattr(snd, 'frequency'): ch = mixer.find_channel()
                        chid = (chid + 1) % nchannels
                        ch = chs[chid]
                        ch.queue(audio)
                except StopIteration:
                    done = True
                    break

                #show_transfer(screen, samples, size=size, type='PAL', axis='imag')
                show_slice(screen, samples, size=size, name=name, antialias=antialias, lines=lines)

                dc = time() - draw_start
                fps = clock.get_fps()
                t = clock.tick_busy_loop(Sampler.videorate)         #cap the framerate
                logger.log(logging.BORING, "Animation: clock tick %d, FPS: %3.3f, drawing took: %.4f", t, fps, dc)
            else:
                logger.debug("Other: %s" % event)

    mixer.quit()
    pygame.quit()

if __name__ == '__main__':
    anim()
