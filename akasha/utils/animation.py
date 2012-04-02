#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import akasha.funct.xoltar.functional as fx
import logging
import numpy as np
import os
import pygame as pg

from fractions import Fraction
from timeit import default_timer as time
from twisted.internet.task import LoopingCall
from twisted.internet import reactor

from .graphing import *
from .math import pcm

from ..audio.generators import Generator
from ..control.io.keyboard import *
from ..timing import sampler, time_slice
from ..tunings import WickiLayout


w = WickiLayout()
VIDEOFRAME = pg.NUMEVENTS - 1

def init_pygame():
    # Set mixer defaults: sampler rate, sample type, number of channels, buffer size
    pg.mixer.pre_init(sampler.rate, pg.AUDIO_S16, 1, 128)
    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
        logger.info("Using %s" % pg.surfarray.get_arraytype().capitalize())
    else:
        raise ImportError('Numpy array package is not installed')

def init_mixer(*args):
    pg.init()
    pg.mixer.quit()
    if isinstance(args, (tuple, list)) and len(args) > 0 and args[0]:
        print args[0]
        pg.mixer.init(*args[0])
    else:
        pg.mixer.init(frequency=sampler.rate, size=-16, channels=1, buffer=128)
    init = pg.mixer.get_init()
    logger.debug("Mixer init: %s sampler rate: %s Video rate: %s" % (init, sampler.rate, sampler.videorate))
    return init

def blit(screen, img):
    pg.surfarray.blit_array(screen, img[:,:,:-1]) # Drop alpha

def show_slice(screen, snd, size=800, name="Resonance", antialias=True, lines=False):
    "Show a slice of the signal"
    snd = snd[0] # FIXME because xoltar uses 'is' to inspect the arguments, snd samples need to be wrapped into a list!
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
    snd = snd[0] # FIXME because xoltar uses 'is' to inspect the arguments, snd samples need to be wrapped into a list!
    img = get_canvas(size)
    tfer = video_transfer(snd, type=type, axis=axis)
    black = (size - tfer.shape[0]) / 2.0
    img[1+black:-black,1:,:] = tfer
    blit(screen, img)
    pg.display.flip()

def change_frequency(snd, key):
    f = w.get(*(pos.get(key, pos[None])))
    snd.frequency = f
    if isinstance(snd, Generator):
        snd.sustain = None
    logger.info("Setting NEW frequency: %r for %s, now at frequency: %s" % (f, snd, snd.frequency))

def clock_tick(snd, it, paint, clock):
    done = False
    events = pg.event.get()
    if len(events) > 1:
        logger.debug("Got %s events to handle: %s" % (len(events), events))
    for event in events:
        if (event.type != VIDEOFRAME):
            done = handle_event(snd, it, event)
            if done:
                break
        else:
            if sampler.paused:
                break
            draw_start = time()
            try:
                samples = it.next()
                audio = pg.sndarray.make_sound(pcm(samples))
                ch = pg.mixer.find_channel()
                ch.queue(audio)
                #logger.debug("Using channel {0} for audio.".format(ch))
            except StopIteration:
                logger.debug("Sound ended!")
                done = True
                break

            paint([samples]) # FIXME wrap samples into a list for xoltar curry to work

            dc = time() - draw_start
            fps = clock.get_fps()
            t = clock.tick_busy_loop(sampler.videorate)
            logger.log(logging.BORING, "Animation: clock tick %d, FPS: %3.3f, drawing took: %.4f", t, fps, dc)
    return done

def handle_event(snd, it, event):
    # Quit
    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
        logger.info("Quitting.")
        return True
    # Pause
    elif (event.type == pg.KEYDOWN and event.key == pg.K_F8) or \
         (event.type == pg.ACTIVEEVENT and event.state == 3):
        sampler.pause()
    # Key down
    elif event.type == pg.KEYDOWN:
        logger.debug("Key down: %s" % event)
        step_size = (5 if event.mod & (pg.KMOD_LSHIFT | pg.KMOD_RSHIFT) else 1)
        # Rewind
        if pg.K_F7 == event.key:
            if isinstance(snd, Generator):
                snd.sustain = None
            logger.info("Rewind")
            it.send('reset')
        # Arrows
        elif pg.K_UP == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                set_timer(sampler.change_frametime(rel = step_size))
            else:
                w.move(-2, 0)
                #w.base *= 2.0
        elif pg.K_DOWN == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                set_timer(sampler.change_frametime(rel = -step_size))
            else:
                w.move(2, 0)
                #w.base /= 2.0
        elif pg.K_LEFT == event.key:
            w.move(0, 1)
        elif pg.K_RIGHT == event.key:
            w.move(0, -1)
        # Change frequency
        elif hasattr(snd, 'frequency'):
            change_frequency(snd, event.key)
            it.send('reset')
    # Key up
    elif (event.type == pg.KEYUP and hasattr(snd, 'frequency')):
        if pg.K_CAPSLOCK == event.key:
            change_frequency(snd, event.key)
            it.send('reset')
        else:
            if isinstance(snd, Generator):
                snd.sustain = it.send('current')[1]
                logger.debug("Key up:   %s, sustain: %s" % (event, snd.sustain))
    else:
        logger.debug("Other: %s" % event)

def set_timer(ms = sampler.frametime):
    pg.time.set_timer(VIDEOFRAME, ms)

def anim(snd, size=800, dur=5.0, name="Resonance", antialias=True, lines=False, init=None):
    """
    Animate complex sound signal
    """
    init_pygame()
    init_mixer(init)
    pg.display.set_caption(name)

    resolution = (size+1, size+1) # FIXME get resolution some other way.
    screen = pg.display.set_mode(resolution, pg.SRCALPHA, 32)

    ch = pg.mixer.find_channel()
    it = iter(snd)

    clock = pg.time.Clock()
    set_timer()

    paint = fx.curry(show_slice, screen, size=size, name=name, antialias=antialias, lines=lines)
    # paint = fx.curry(show_transfer, screen, size=size, type='PAL', axis='imag')

    done = False
    while not done:
        done = clock_tick(snd, it, paint, clock)

    it.close()
    del it
    pg.mixer.quit()
    pg.quit()


if __name__ == '__main__':
    init_pygame()

