#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import numpy as np
import os
import pygame as pg

from fractions import Fraction
from timeit import default_timer as time

from .graphing import *
from .math import pcm

from ..audio.generators import Generator
from ..control.io.keyboard import *
from ..funct import blockwise2
from ..timing import Sampler, time_slice
from ..tunings import WickiLayout


w = WickiLayout()

def init_pygame():
    # Set mixer defaults: Sampler rate, sample type, number of channels, buffer size
    pg.mixer.pre_init(Sampler.rate, pg.AUDIO_S16, 1, 128)
    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
        logger.info("Using %s" % pg.surfarray.get_arraytype().capitalize())
    else:
        raise ImportError('Numpy array package is not installed')

def init_mixer(*args):
    pg.init()
    pg.mixer.quit()
    print args[0]
    if args[0]:
        pg.mixer.init(*args[0])
    else:
        pg.mixer.init(frequency=Sampler.rate, size=-16, channels=1, buffer=128)
    init = pg.mixer.get_init()
    logger.debug("Mixer init: %s Sampler rate: %s Video rate: %s" % (init, Sampler.rate, Sampler.videorate))
    return init

def blit(screen, img):
    pg.surfarray.blit_array(screen, img[:,:,:-1]) # Drop alpha

def show_slice(screen, snd, size=800, name="Resonance", antialias=True, lines=False):
    "Show a slice of the signal"
    if lines:
        img = get_canvas(size)
        blit(screen, img)
        img = draw(snd, size, antialias=antialias, lines=lines, screen=screen, img=img)
    else:
        img = draw(snd, size, antialias=antialias, lines=lines, screen=screen)
        blit(screen, img)
    pg.display.flip()

def show_transfer(screen, snd, size=720, name="Transfer", type='PAL', axis='imag'):
    "Show a slice of the signal"
    img = get_canvas(size)
    tfer = video_transfer(snd, type=type, axis=axis)
    black = (size - tfer.shape[0]) / 2.0
    img[1+black:-black,1:,:] = tfer
    blit(screen, img)
    pg.display.flip()

def anim(snd, size=800, dur=5.0, name="Resonance", antialias=True, lines=False, init=None):
    """
    Animate complex sound signal
    """
    init_pygame()
    init_mixer(init)
    pg.display.set_caption(name)

    clock = pg.time.Clock()
    resolution = (size+1, size+1) # FIXME get resolution some other way. This was: img.shape[:2]
    screen = pg.display.set_mode(resolution, pg.SRCALPHA, 32) # flags=pygame.SRCALPHA, depth=32

    chs = []
    if hasattr(snd, 'frequency'):
        nchannels = 1
    else:
        nchannels = 1
    for i in xrange(nchannels):
        chs.append(pg.mixer.find_channel())
    chid = 0
    ch = chs[chid]
    
    def blockiter(snd):
        return (iter(snd) if isinstance(snd, Generator) else blockwise2(snd, 1, Sampler.blocksize()))
    it = blockiter(snd)

    VIDEOFRAME = pg.NUMEVENTS - 1
    def set_timer():
        # FIXME - complain about ints to pygame
        ms = int(round(1000.0 / Sampler.videorate)) # 40 ms for 25 Hz
        pg.time.set_timer(VIDEOFRAME, ms)
    set_timer()

    Sampler.paused = False
    def pause():
        Sampler.paused = not Sampler.paused
        logger.info("Pause" if Sampler.paused else "Play")

    def change_frequency(snd, key):
        f = w.get(*(pos.get(key, pos[None])))
        snd.frequency = f
        if isinstance(snd, Generator):
            snd.sustain = None
        logger.info("Setting NEW frequency: %r for %s, now at frequency: %s" % (f, snd, snd.frequency))

    done = False
    while not done:
        events = pg.event.get()
        if len(events) > 1:
            logger.debug("Got %s events to handle: %s" % (len(events), events))
        for event in events:

            # Quit
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == 27):
                done = True
                break

            # Pause
            elif (event.type == pg.KEYDOWN and event.key == pg.K_F8) or \
                 (event.type == pg.ACTIVEEVENT and event.state == 3):
                pause()

            # -- Key downs --
            elif event.type == pg.KEYDOWN:
                logger.debug("Key down: %s" % event)
                # Rewind
                if pg.K_F7 == event.key:
                    it = blockiter(snd)
                    if isinstance(snd, Generator):
                        snd.sustain = None
                    logger.info("Rewind")
                # Arrows
                elif pg.K_UP == event.key:
                    # Sampler.videorate += 1
                    # set_timer()
                    #w.move(-2, 0)
                    w.base *= 2.0
                elif pg.K_DOWN == event.key:
                    # Sampler.videorate = max(Sampler.videorate - 1, 1) # Prevent zero division
                    # set_timer()
                    #w.move(2, 0)
                    w.base /= 2.0
                elif pg.K_LEFT == event.key:
                    w.move(0, 1)
                elif pg.K_RIGHT == event.key:
                    w.move(0, -1)
                # Change frequency
                elif hasattr(snd, 'frequency'):
                    change_frequency(snd, event.key)
                    it = blockiter(snd)

            # -- Key ups --
            elif (event.type == pg.KEYUP and hasattr(snd, 'frequency')):
                if pg.K_CAPSLOCK == event.key:
                    change_frequency(snd, event.key)
                    it = blockiter(snd)
                else:
                    if isinstance(snd, Generator):
                        #snd.sustain = np.index(snd, it.next()[0]) - blocksize() # FIXME #asl.start
                        pass
                    logger.debug("Key up:   %s, sustain: %s" % (event, snd.sustain))

            # Video frame
            elif (event.type == VIDEOFRAME):
                if Sampler.paused:
                    break
                
                draw_start = time()
                try:
                    samples = it.next()
                    audio = pg.sndarray.make_sound(pcm(samples))
                    chid = (chid + 1) % nchannels
                    ch = chs[chid]
                    ch.queue(audio)
                except StopIteration:
                    logger.debug("Sound ended!")
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

    it.close()
    del it
    pg.mixer.quit()
    pg.quit()

if __name__ == '__main__':
    init_pygame()
