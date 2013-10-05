#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'twisted.internet.reactor' has no 'run/stop/running' member
"""
Animation module
"""

from __future__ import division

import logging
import numpy as np
import pygame as pg
import sys
import time
import traceback

from timeit import default_timer as timer
from twisted.internet.task import LoopingCall
from twisted.internet import reactor

from akasha.audio.generators import Generator
from akasha.control.io.keyboard import pos
from akasha.funct import blockwise
from akasha.graphic.drawing import get_canvas, blit, draw, video_transfer
from akasha.timing import sampler
from akasha.tunings import PianoLayout, WickiLayout
from akasha.utils import issequence
from akasha.utils.math import pcm, minfloat
from akasha.utils.log import logger



# import sys

# def trace(frame, event, arg):
#     if event == 'c_call' or arg is not None and 'IPython' not in frame.f_code.co_filename:
#         print "%s, %s: %d" % (event, frame.f_code.co_filename, frame.f_lineno)
#     return trace

# sys.settrace(trace)


w = WickiLayout()
# w = PianoLayout()

VIDEOFRAME = pg.NUMEVENTS - 1
AUDIOFRAME = pg.NUMEVENTS - 2


def anim(snd, size=800, name='Resonance', antialias=True, lines=False, colours=True,
         mixer_options=(), loop='pygame', style='complex'):
    """
    Animate complex sound signal
    """
    logger.info(
        "Akasha animation is using %s Hz sampler rate and %s fps video rate." %
        (sampler.rate, sampler.videorate))

    sampler.paused = False
    screen = init_pygame(name, size)
    channel = init_mixer(*mixer_options)

    if style == 'complex':
        paint_fn = lambda snd: show_slice(
            screen, snd, size=size,
            antialias=antialias, lines=lines, colours=colours
            )
    elif style == 'transfer':
        paint_fn = lambda snd: show_transfer(screen, snd, size=size, standard='PAL', axis='imag')
    else:
        logger.err("Unknown animation style: '{0}'".format(style))
        cleanup()

    # set_timer(AUDIOFRAME, int(round(sampler.frametime / 5)))
    set_timer(VIDEOFRAME, sampler.frametime)

    if loop == 'pygame':
        pygame_loop(snd, channel, paint_fn)
    elif loop == 'twisted':
        twisted_loop(snd, channel, paint_fn)
    else:
        logger.err("Unknown event loop: '{0}'".format(loop))
        cleanup()


def pygame_loop(snd, channel, paint_fn):
    clock = pg.time.Clock()
    it = blockwise(snd, sampler.blocksize())

    while True:
        try:
            timestamp = timer()

            (input_time, audio_time, video_time) = handle_events(snd, it, channel, paint_fn)

            loop_time = timer() - timestamp

            fps = clock.get_fps()
            percent = loop_time / (1.0 / sampler.videorate) * 100
            av_percent = (audio_time + video_time) / (1.0 / sampler.videorate) * 100
            t = clock.tick_busy_loop(sampler.videorate)

            if not sampler.paused:
                logger.log(logging.BORING,
                           "Animation: clock tick %d, FPS: %3.3f, loop: %.4f, (%.2f %%), input: %.6f, audio: %.6f, video: %.4f, (%.2f %%)", t, fps, loop_time, percent, input_time, audio_time, video_time, av_percent)
        except KeyboardInterrupt, err:
            # See http://stackoverflow.com/questions/2819931/handling-keyboardinterrupt-when-working-with-pygame
            logger.info("Got KeyboardInterrupt (CTRL-C)!".format(type(err)))
            cleanup(it)
            break
        except SystemExit, err:
            logger.info("Ending animation: %s" % err.message)
            cleanup(it)
            break
        except Exception, err:
            try:
                exc = sys.exc_info()[:2]
                logger.error("Unexpected exception %s: %s\n%s" % (exc[0], exc[1], traceback.format_exc()))
            finally:
                del exc
                cleanup(it)
                break


def twisted_loop(snd, it, channel, paint_fn):
    # See: http://bazaar.launchpad.net/~game-hackers/game/trunk/view/head:/game/view.py

    pg.display.init()
    it = blockwise(snd, sampler.blocksize())

    # renderCall = LoopingCall(do_audio_video, it, channel, paint_fn)
    # renderdef = renderCall.start(1 / sampler.videorate, now=False)
    # renderdef.addErrback(handle_error)

    inputCall = LoopingCall(handle_events, snd, it, channel, paint_fn)
    finished = inputCall.start(1 / (sampler.videorate * 2), now=False)
    finished.addErrback(handle_error)

    # finished.addCallback(lambda ign: renderCall.stop())
    # finished.addCallback(lambda ign: cleanup())

    if not reactor.running:
        reactor.run()  # pylint: disable=E1101


def handle_events(snd, it, channel, paint_fn):
    """
    Event handling dispatcher.
    """
    reset = False
    input_time, audio_time, video_time = 0, 0, 0

    events = pg.event.get()

    inputs = [event for event in events if event.type not in (AUDIOFRAME, VIDEOFRAME)]
    videoframes = [event for event in events if event.type == VIDEOFRAME]

    # input_start = timer()
    reset = handle_inputs(snd, it, inputs)
    # input_time = timer() - input_start

    # Paint
    if videoframes and not sampler.paused:
        try:
            if reset:
                try:
                    samples = it.send('reset')
                except TypeError:
                    samples = it.next()
            else:
                samples = it.next()
            # logger.debug("iterator on: %s" % (it.send('current'),))

            audio_start = timer()
            queue_audio(samples, channel)
            audio_time = timer() - audio_start

            video_start = timer()
            paint_fn(samples)
            video_time = timer() - video_start

            drop_frames(videoframes, it)
        except StopIteration:
            raise SystemExit('Sound ended!')

    if reactor.running:
        # reactor.stop()  # pylint: disable=E1101
        pass

    return (input_time, audio_time, video_time)


def queue_audio(samples, channel):
    """
    Queue samples into a mixer channel.
    """
    channel.queue(pg.sndarray.make_sound(pcm(samples)))


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
    if len(frames) > 1:
        drop = len(frames) - 1
        for i in xrange(drop):
            try:
                it.next()
            except StopIteration:
                raise SystemExit('Sound ended when dropping frames!')
        logger.warn("Dropped %s frames!" % drop)


def handle_inputs(snd, it, inputs):
    if len(inputs) == 0:
        return False

    reset = False

    input_start = timer()
    for event in inputs:
        # timestamp = timer()
        # logger.debug("Event: %s at %s" % (event, timestamp))

        reset |= handle_input(snd, it, event)

        # input_time = timer() - timestamp
        # if input_time > 0.001:
        #     logger.warn("*** SLOW Input events handling: %.4f", input_time)

    logger.debug("Inputs: %s handled in %.4f seconds." % (len(inputs), timer() - input_start))

    return reset


def handle_input(snd, it, event):
    """
    Handle pygame events.
    """
    # Quit
    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
        raise SystemExit('Quit.')
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
            it.send('reset')
            return True
        # Arrows
        elif pg.K_UP == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                set_timer(ms = sampler.change_frametime(rel=step_size))
            else:
                # w.move(-2, 0)
                w.base *= 2.0
        elif pg.K_DOWN == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                set_timer(ms = sampler.change_frametime(rel=-step_size))
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
            return True
    # Key up
    elif (event.type == pg.KEYUP and hasattr(snd, 'frequency')):
        if pg.K_CAPSLOCK == event.key:
            change_frequency(snd, event.key, it)
            return True
        else:
            if isinstance(snd, Generator):
                try:
                    snd.sustain = it.send('current')[0]
                except TypeError:
                    snd.sustain = 0
                logger.debug("Key '%s' (%s) up, sustain: %s" % (pg.key.name(event.key), event.key, snd.sustain))
                return True
    # Mouse
    elif hasattr(snd, 'frequency') and event.type in (
        pg.MOUSEBUTTONDOWN,
        pg.MOUSEBUTTONUP,
        pg.MOUSEMOTION
    ):
        if (event.type == pg.MOUSEBUTTONDOWN):
            snd.pitch_bend = 0
        elif (event.type == pg.MOUSEBUTTONUP):
            snd.pitch_bend = None
        elif (event.type == pg.MOUSEMOTION) and getattr(snd, 'pitch_bend', None) is not None:
            size = pg.display.get_surface().get_size()[1] or 0
            freq = snd.frequency
            snd.pitch_bend = event.pos[1]

            logger.debug("Pitch bend == event.pos[0]: %s" % snd.pitch_bend)

            odd = (size + 1) % 2
            norm_size = (size // 2) * 2 + odd
            scale = np.logspace(-1 / 4, 1 / 4, norm_size, endpoint=True, base=2)
            ratio = scale[snd.pitch_bend]

            new_freq = np.clip(
                snd.frequency * ratio,
                a_min=-minfloat()[0],
                a_max=sampler.nyquist
            )
            snd.frequency = new_freq
            logger.info(
                "Pitched frequency to %s (with ratio %.04f) position %s, bend %s. [%s, %s, %s]" %
                (
                    snd.frequency, ratio, event.pos[1], snd.pitch_bend,
                    scale[0], scale[len(scale)//2], scale[-1]
                ))
    else:
        if event.type != AUDIOFRAME:
            logger.debug("Other: %s" % event)
    return False


def init_pygame(name="Resonance", size=800):
    """
    Initialize Pygame mixer settings and surface array.
    """
    pg.quit()

    logger.info(
        "Pygame initialized with %s loaded modules (%s failed)." %
        pg.init())

    screen = init_display(name, size)
    surface = pg.display.get_surface()
    logger.info("Inited display %s with flags: %s" % (screen, surface.get_flags()))

    return screen


def init_display(name, size):
    """
    Initialize Pygame display and surface arrays.
    Returns Pygame screen.
    """
    pg.display.quit()

    flags = 0
    flags |= pg.SRCALPHA
    flags |= pg.HWSURFACE
    # flags |= pg.OPENGL
    # flags |= pg.DOUBLEBUF

    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')

    try:
        # FIXME get resolution some other way.
        mode = pg.display.set_mode((size, size), flags, 32)
        pg.display.init()
    except Exception, err:
        logger.error("Something bad happened on init_display(): %s" % err)

    return mode


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


def set_timer(event=VIDEOFRAME, ms=sampler.frametime):
    """
    Set and start pygame timer interval for VIDEOFRAME events.
    """
    pg.time.set_timer(event, ms)


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
    pg.mixer.quit()
    pg.display.quit()
    pg.quit()
    if it:
        if hasattr(it, 'close'):
            it.close()
        del it
    logger.info("Done cleanup.")


def change_frequency(snd, key, it):
    """
    Change frequency of the sound based on key position.
    """
    snd.frequency = w.get(*(pos.get(key, pos[None])))

    if isinstance(snd, Generator):
        snd.sustain = None

    logger.info("Changed frequency: %s." % snd)

