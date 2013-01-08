#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'twisted.internet.reactor' has no 'run/stop/running' member
"""
Animation module
"""

from __future__ import division

import logging
import pygame as pg
import time

from funckit import xoltar as fx
from timeit import default_timer as timer
from twisted.internet.task import LoopingCall
from twisted.internet import reactor

from akasha.audio.generators import Generator
from akasha.control.io.keyboard import pos
from akasha.graphic.drawing import get_canvas, draw, video_transfer
from akasha.timing import sampler
from akasha.tunings import WickiLayout
from akasha.utils.math import pcm
from akasha.utils.log import logger


w = WickiLayout()
VIDEOFRAME = pg.NUMEVENTS - 1


def change_frequency(snd, key):
    """
    Change frequency of the sound based on key position.
    """
    f = w.get(*(pos.get(key, pos[None])))
    snd.frequency = f
    if isinstance(snd, Generator):
        snd.sustain = None
    logger.info(
        "Setting NEW frequency: %r for %s, now at frequency: %s" %
        (f, snd, snd.frequency))


def handle_input(snd, it, event):
    """
    Handle pygame events.
    """
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
                set_timer(sampler.change_frametime(rel=step_size))
            else:
                # w.move(-2, 0)
                w.base *= 2.0
        elif pg.K_DOWN == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                set_timer(sampler.change_frametime(rel=-step_size))
            else:
                # w.move(2, 0)
                w.base /= 2.0
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
                snd.sustain = it.send('current')[0]
                logger.debug("Key up:   %s, sustain: %s" % (event, snd.sustain))
    else:
        logger.debug("Other: %s" % event)
    return False


def paint_frame(it, ch, paint_fn, clock):
    """
    Queue audio and paint a frame.
    """
    done = False
    if sampler.paused:
        return done
    draw_start = timer()
    try:
        samples = it.next()
        audio = pg.sndarray.make_sound(pcm(samples))
        ch.queue(audio)
        paint_fn([samples])  # FIXME wrap samples into a list for xoltar curry to work
    except StopIteration:
        logger.debug("Sound ended!")
        done = True
        return done

    dc = timer() - draw_start
    fps = clock.get_fps()
    t = clock.tick_busy_loop(sampler.videorate)
    logger.log(
        logging.BORING,
        "Animation: clock tick %d, FPS: %3.3f, drawing took: %.4f", t, fps, dc)
    return done


def handle_events(snd, it, ch, paint_fn, clock, twisted_loop=False):
    """
    Event handling dispatcher.
    """
    done = False
    input_start = timer()
    events = pg.event.get()
    for event in events:
        if twisted_loop or (event.type != VIDEOFRAME):
            done = handle_input(snd, it, event)
        if twisted_loop or (event.type == VIDEOFRAME):
            done |= paint_frame(it, ch, paint_fn, clock)
        if done:
            break
    dc = timer() - input_start
    if len(events) > 1:
        logger.debug("Handled %s events in %.4f seconds: %s" % (len(events), dc, events))
    if done and reactor.running:  # pylint: disable=E1101
        pg.display.quit()
        reactor.stop()  # pylint: disable=E1101
    return done


def init_pygame():
    """
    Initialize Pygame mixer settings and surface array.
    """
    # Set mixer defaults: sampler rate, sample type, number of channels, buffer size
    pg.mixer.pre_init(sampler.rate, pg.AUDIO_S16, 1, 128)
    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
        logger.info("Using %s" % pg.surfarray.get_arraytype().capitalize())
    else:
        raise ImportError('Numpy array package is not installed')


def init_mixer(*args):
    """
    Initialize the Pygame mixer.
    """
    pg.init()
    pg.mixer.quit()
    if isinstance(args, (tuple, list)) and len(args) > 0 and args[0]:
        print args[0]
        pg.mixer.init(*args[0])
    else:
        pg.mixer.init(frequency=sampler.rate, size=-16, channels=1, buffer=128)
    init = pg.mixer.get_init()
    logger.debug(
        "Mixer init: %s sampler rate: %s Video rate: %s" %
        (init, sampler.rate, sampler.videorate))
    return init


def blit(screen, img):
    """
    Blit the screen.
    """
    pg.surfarray.blit_array(screen, img[:, :, :-1])  # Drop alpha


def show_slice(screen, snd, size=800, antialias=True, lines=False, colours=True):
    """
    Show a sound signal on screen.
    """
    # FIXME because xoltar uses 'is' to inspect the arguments,
    # snd samples need to be wrapped into a list!
    snd = snd[0]
    if lines and antialias:  # Using Pygame drawing, so blit before
        img = get_canvas(size)
        blit(screen, img)
        img = draw(snd, size, antialias=antialias, lines=lines, colours=colours,
                   screen=screen, img=img)
    else:
        img = draw(snd, size, antialias=antialias, lines=lines, colours=colours, screen=screen)
        blit(screen, img)
    pg.display.flip()


def show_transfer(screen, snd, size=720, standard='PAL', axis='imag'):
    """
    Show a sound signal using the old video tape audio recording technique.
    See: http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """
    # FIXME because xoltar uses 'is' to inspect the arguments,
    # snd samples need to be wrapped into a list!
    snd = snd[0]
    img = get_canvas(size)
    tfer = video_transfer(snd, standard=standard, axis=axis, horiz=size)
    black = (size - tfer.shape[0]) / 2.0
    img[1 + black:-black, 1:, :] = tfer
    blit(screen, img)
    pg.display.flip()


def set_timer(ms=sampler.frametime):
    """
    Set and start pygame timer interval for VIDEOFRAME events.
    """
    pg.time.set_timer(VIDEOFRAME, ms)


def handleError(err):
    """
    Logging error handler.
    """
    logger.error("%s happened!" % str(err))
    pg.display.quit()
    reactor.stop()  # pylint: disable=E1101
    raise err


def anim(snd, size=800, name="Resonance", antialias=True, lines=False, colours=True,
         init=None, loop='pygame'):
    """
    Animate complex sound signal
    """
    init_pygame()
    init_mixer(init)
    pg.display.set_caption(name)

    resolution = (size, size)  # FIXME get resolution some other way.
    screen = pg.display.set_mode(resolution, pg.SRCALPHA, 32)

    ch = pg.mixer.find_channel()
    it = iter(snd)

    paint_fn = lambda snd: show_slice(
        screen,
        snd,
        size=size,
        antialias=antialias,
        lines=lines,
        colours=colours
    )
    # paint_fn = fx.curry_function(show_slice, screen, size=size,
    #                     antialias=antialias, lines=lines, colours=colours)
    #paint_fn = fx.curry_function(show_transfer, screen, size=size, standard='PAL', axis='imag')

    clock = pg.time.Clock()

    if loop == 'pygame':
        set_timer()
        done = False
        while not done:
            done = handle_events(snd, it, ch, paint_fn, clock)
            time.sleep(1 / 1000)  # Fixme: This reduces calls to handle_events, but is it necessary?
    else:
        pg.display.init()

        renderCall = LoopingCall(paint_frame, it, ch, paint_fn, clock)
        inputCall = LoopingCall(handle_events, snd, it, ch, paint_fn, clock)

        renderdef = renderCall.start(1 / sampler.videorate, now=False)
        renderdef.addErrback(handleError)

        finished = inputCall.start(1 / sampler.videorate / 10, now=False)
        finished.addErrback(handleError)

        finished.addCallback(lambda ign: renderCall.stop())
        finished.addCallback(lambda ign: pg.display.quit())
        reactor.run()  # pylint: disable=E1101

    it.close()
    del it
    pg.mixer.quit()
    pg.quit()


class SignalView(object):
    """
    A view for animated sound signal and audio output.
    """
    def __init__(self, position=(0, 0), size=(800, 800), antialias=True, lines=False):
        self.position = position
        self.size = size
        self.antialias = antialias
        self.lines = lines


class Window(object):
    """
    Pygame based window.
    """
    def __init__(
        self,
        clock=reactor,
        display=pg.display,
        event=pg.event,
        view=SignalView,
        size=(800, 800),
        name="Resonance"
    ):
        self.clock = clock
        self.display = display
        self.event = event
        self.size = size
        self.viewType = view
        self._view = view(size=size)
        pg.display.set_caption(name)

    def go(self, snd):
        """
        Go live with the Window.
        """
        self.display.init()
        # self.display.set_caption(name)

        renderCall = LoopingCall(handle_events, (snd, iter(snd), paint_frame, self.clock))
        renderdef = renderCall.start(1 / sampler.videorate, now=False)
        renderdef.addCallback(lambda ign: self.display.quit())
        renderdef.addErrback(handleError)

        reactor.run()  # pylint: disable=E1101
