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

from timeit import default_timer as timer
from twisted.internet.task import LoopingCall
from twisted.internet import reactor

from akasha.audio.generators import Generator
from akasha.control.io.keyboard import pos
from akasha.graphic.drawing import get_canvas, blit, draw, video_transfer
from akasha.timing import sampler
from akasha.tunings import WickiLayout
from akasha.utils import issequence
from akasha.utils.math import pcm
from akasha.utils.log import logger


w = WickiLayout()
VIDEOFRAME = pg.NUMEVENTS - 1


def anim(snd, size=800, name="Resonance", antialias=True, lines=False, colours=True,
         mixer_options=(), loop='pygame'):
    """
    Animate complex sound signal
    """
    screen, ch = init_pygame(name, size, mixer_options)

    it = iter(snd)

    paint_fn = lambda snd: show_slice(
        screen, snd, size=size,
        antialias=antialias, lines=lines, colours=colours
    )
    # paint_fn = lambda snd: show_transfer(screen, snd, size=size, standard='PAL', axis='imag')

    clock = pg.time.Clock()

    set_timer()
    if loop == 'pygame':
        done = False
        while not done:
            done = handle_events(snd, it, ch, paint_fn, clock)
            time.sleep(1 / 1000)  # Fixme: This reduces calls to handle_events, but is it necessary?
        cleanup(it)
    else:
        # See: http://bazaar.launchpad.net/~game-hackers/game/trunk/view/head:/game/view.py

        pg.display.init()

        # renderCall = LoopingCall(do_audio_video, it, ch, paint_fn)
        # renderdef = renderCall.start(1 / sampler.videorate, now=False)
        # renderdef.addErrback(handle_error)

        inputCall = LoopingCall(handle_events, snd, it, ch, paint_fn, clock)
        finished = inputCall.start(1 / (sampler.videorate * 2), now=False)
        finished.addErrback(handle_error)

        # finished.addCallback(lambda ign: renderCall.stop())
        # finished.addCallback(lambda ign: cleanup())

        if not reactor.running:
            reactor.run()  # pylint: disable=E1101


def init_pygame(name="Resonance", size=800, mixer_options=()):
    """
    Initialize Pygame mixer settings and surface array.
    """
    logger.info(
        "Akasha is using %s Hz sampler rate and %s fps video rate." %
        (sampler.rate, sampler.videorate))

    screen = init_display(name, size)
    channel = init_mixer(*mixer_options)

    logger.info(
        "Pygame initialized with %s loaded modules (%s failed)." %
        pg.init())

    return screen, channel


def init_display(name, size):
    """
    Initialize Pygame display and surface arrays.
    Returns Pygame screen.
    """
    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')

    pg.display.set_caption(name)
    resolution = (size, size)  # FIXME get resolution some other way.

    return pg.display.set_mode(resolution, pg.SRCALPHA, 32)


def init_mixer(*args):
    """
    Initialize the Pygame mixer.
    """
    pg.mixer.quit()

    # Set mixer defaults: sample rate, sample size, number of channels, buffer size
    if issequence(args) and 0 < len(args) <= 3:
        pg.mixer.init(*args)
    else:
        pg.mixer.init(frequency=sampler.rate, size=-16, channels=1, buffer=512)

    logger.info(
        "Mixer has %s Hz sample rate with %s size samples and %s channels." %
        pg.mixer.get_init())

    return pg.mixer.find_channel()


def set_timer(ms=sampler.frametime):
    """
    Set and start pygame timer interval for VIDEOFRAME events.
    """
    pg.time.set_timer(VIDEOFRAME, ms)


def handle_error(err):
    """
    Logging error handler.
    """
    logger.error("Error traceback:\n%s" % str(err))
    if reactor.running:
        reactor.stop()  # pylint: disable=E1101
    cleanup()
    raise err


def cleanup(it=None):
    """
    Clean up: Quit pygame, close iterator.
    """
    logger.info("Clean up: Quit pygame, close iterator.")
    pg.mixer.quit()
    pg.display.quit()
    pg.quit()
    if it:
        it.close()
        del it
    logger.info("Done cleanup.")


def handle_events(snd, it, ch, paint_fn, clock):
    """
    Event handling dispatcher.
    """
    done = False
    start = timer()
    events = pg.event.get()

    inputs = [event for event in events if event.type != VIDEOFRAME]
    frames = [event for event in events if event.type == VIDEOFRAME]

    # Handle input
    if inputs:
        input_start = timer()
        for event in inputs:
            done = handle_input(snd, it, event)
            if done:
                break
        logger.debug("Inputs: %s handled in %.4f seconds." % (len(inputs), timer() - input_start))

    # Paint
    if frames and not sampler.paused and not done:
        draw_start = timer()

        done |= do_audio_video(it, ch, paint_fn)

        dc = timer() - draw_start
        fps = clock.get_fps()
        t = clock.tick_busy_loop(sampler.videorate)

        logger.log(logging.BORING,
            "Animation: clock tick %d, FPS: %3.3f, drawing took: %.4f", t, fps, dc)

        done |= drop_frames(frames, it)

    if len(events) - len(inputs) > 1:
        logger.info("Events: %s handled in %.4f seconds." % (len(events), timer() - start))

    if reactor.running and done:
        reactor.stop()  # pylint: disable=E1101
        cleanup()
    else:
        return done


def do_audio_video(it, ch, paint_fn):
    if not sampler.paused:
        samples = next_block(it)
        if samples is not None:
            queue_audio(samples, ch)
            paint_fn(samples)
        else:
            return True  # done
    return False


def next_block(it):
    """
    Get a next block of samples.
    """
    try:
        return it.next()
    except StopIteration:
        logger.debug("Sound ended!")
        return None


def queue_audio(samples, ch):
    """
    Queue samples into a mixer channel.
    """
    ch.queue(pg.sndarray.make_sound(pcm(samples)))


def show_slice(screen, snd, size=800, antialias=True, lines=False, colours=True):
    """
    Show a sound signal on screen.
    """
    img = draw(snd, size,
               antialias=antialias, lines=lines, colours=colours,
               axis=True, screen=screen)

    blit(screen, img)
    pg.display.flip()


def show_transfer(screen, snd, size=720, standard='PAL', axis='imag'):
    """
    Show a sound signal using the old video tape audio recording technique.
    See: http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """
    img = get_canvas(size)
    tfer = video_transfer(snd, standard=standard, axis=axis, horiz=size)

    black = (size - tfer.shape[0]) / 2.0
    img[:, black:-black, :] = tfer[:, :img.shape[1], :].transpose(1, 0, 2)

    blit(screen, img)
    pg.display.flip()


def drop_frames(frames, it):
    """
    Drop the number of frame events over one from the iterator.
    """
    done = False
    if len(frames) > 1:
        drop = len(frames) - 1
        for i in xrange(drop):
            try:
                it.next()
            except StopIteration:
                done = True
                break
        logger.warn("Dropped %s frames!" % drop)
    return done


def handle_input(snd, it, event):
    """
    Handle pygame events.
    """
    # Quit
    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
        logger.info("Quitting.")
        return True
    # Pause
    elif (event.type in [pg.KEYDOWN, pg.KEYUP] and event.key == pg.K_F8) or \
         (event.type == pg.ACTIVEEVENT and event.state == 3):
        if event.type is not pg.KEYUP:
            sampler.pause()
        return False
    # Key down
    elif event.type == pg.KEYDOWN:
        logger.debug("Key '%s' (%s) down." % (pg.key.name(event.key), event.key))
        step_size = (5 if event.mod & (pg.KMOD_LSHIFT | pg.KMOD_RSHIFT) else 1)
        # Rewind
        if pg.K_F7 == event.key:
            if isinstance(snd, Generator):
                snd.sustain = None
            logger.info("Rewind")
            reset_iterator(it)
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
            change_frequency(snd, event.key, it)
    # Key up
    elif (event.type == pg.KEYUP and hasattr(snd, 'frequency')):
        if pg.K_CAPSLOCK == event.key:
            change_frequency(snd, event.key, it)
        else:
            if isinstance(snd, Generator):
                snd.sustain = it.send('current')[0]
                logger.debug("Key '%s' (%s) up, sustain: %s" % (pg.key.name(event.key), event.key, snd.sustain))
    else:
        logger.debug("Other: %s" % event)

    return False


def reset_iterator(it):
    """Reset an iterator and catch possible TypeErrors if it has just started."""
    try:
        it.send('reset')
    except TypeError, err:
        logger.warn(err)


def change_frequency(snd, key, it):
    """
    Change frequency of the sound based on key position.
    """
    snd.frequency = w.get(*(pos.get(key, pos[None])))

    if isinstance(snd, Generator):
        snd.sustain = None

    logger.info("Changed frequency: %s." % snd)
    reset_iterator(it)
