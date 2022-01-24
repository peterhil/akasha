#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Animation module
"""

from __future__ import division

import numpy as np
import pygame as pg

from akasha.audio.mixins.releasable import Releasable
from akasha.graphic.drawing import get_canvas
from akasha.graphic.widgets import ComplexView, VideoTransferView
from akasha.math import div_safe_zero, pcm, minfloat
from akasha.settings import config
from akasha.timing import sampler, Timed, Watch
from akasha.tunings import PianoLayout, WickiLayout
from akasha.utils import issequence
from akasha.utils.log import logger

keyboard = WickiLayout()
# keyboard = PianoLayout()


def anim(snd, size=800, name='Resonance', antialias=True, lines=False, colours=True,
         mixer_options=(), style='complex'):
    """
    Animate complex sound signal
    """
    logger.info(
        "Akasha animation is using %s Hz sampler rate and %s fps video rate.",
        sampler.rate, sampler.videorate
    )

    sampler.paused = True if hasattr(snd, 'frequency') else False
    screen = init_pygame(name, size)
    channel = init_mixer(*mixer_options)

    if style == 'complex':
        widget = ComplexView(screen, size, antialias=antialias, lines=lines, colours=colours)
    elif style == 'transfer':
        widget = VideoTransferView(screen, size=size, standard='PAL', axis='real')
    else:
        logger.error("Unknown animation style: '%s'", style)
        cleanup()
        return False

    return loop(snd, channel, widget)


def loop(snd, channel, widget):
    clock = pg.time.Clock()
    t = clock.tick_busy_loop(sampler.videorate)
    watch = Watch()
    last = 0

    # Handle first time, draw axis and bg
    first_time = True
    widget.render(np.array([0, 0, 0], dtype=np.complex128))
    pg.display.flip()

    while True:
        try:
            with Timed() as loop_time:
                input_time, audio_time, video_time = 0, 0, 0

                with Timed() as input_time:
                    for event in pg.event.get():
                        if handle_input(snd, watch, event):
                            if first_time and hasattr(snd, 'frequency') and sampler.paused:
                                sampler.pause()
                            watch.reset()
                            last = 0
                if not sampler.paused:
                    current = watch.next()
                    current_slice = slice(sampler.at(last, np.int), sampler.at(current, np.int))
                    samples = snd[current_slice]

                    if len(samples) == 0 and current_slice.start != current_slice.stop:
                        raise SystemExit('Sound ended!')
                    if len(samples) > 0:
                        with Timed() as audio_time:
                            queue_audio(samples, channel)
                        with Timed() as video_time:
                            widget.render(samples)
                        pg.display.flip()
                        last = current
            if not sampler.paused:
                percent = float(loop_time) / (1.0 / sampler.videorate) * 100
                av_percent = (float(audio_time) + float(video_time)) / float(loop_time) * 100
                fps = watch.get_fps()
                if percent >= config.logging_limits.LOOP_THRESHOLD_PERCENT:
                    logger.warning(
                        "Anim: (%d) %.1f%% %.1f fps,\tloop: %.1fhz "
                        "(%.1f%%) video: %.1fhz audio: %.1fkhz io: %.1fkhz", t, percent, fps, div_safe_zero(1, loop_time),
                        av_percent, div_safe_zero(1, video_time), div_safe_zero(1e-3, audio_time), div_safe_zero(1e-3, input_time)
                    )
            t = clock.tick_busy_loop(sampler.videorate)
        except KeyboardInterrupt:
            # See http://stackoverflow.com/questions/2819931/handling-keyboardinterrupt-when-working-with-pygame
            logger.info("Got KeyboardInterrupt (CTRL-C)!")
            break
        except SystemExit as err:
            logger.info("Ending animation: %s", err)
            break
    cleanup()
    if isinstance(snd, Releasable):
        snd.release_at(None)
    return watch


def queue_audio(samples, channel):
    """
    Queue samples into a mixer channel.
    """
    channel.queue(pg.sndarray.make_sound(pcm(samples)))


def key_pause(event):
    return (event.type == pg.KEYDOWN and event.key == pg.K_F8) \
        or (event.type == pg.ACTIVEEVENT and event.state == 3)


def key_escape(event):
    return (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE)


def handle_input(snd, watch, event):
    """
    Handle pygame events.
    """
    # Quit
    if event.type == pg.QUIT or key_escape(event):
        raise SystemExit('Quit.')

    # Pause
    if key_pause(event):
        sampler.pause()
        watch.pause()
        return

    # Key down
    elif event.type == pg.KEYDOWN:
        # logger.debug("Key '%s' (%s) down.", pg.key.name(event.key), event.key)
        step_size = (5 if event.mod & (pg.KMOD_LSHIFT | pg.KMOD_RSHIFT) else 1)
        # Rewind
        if pg.K_F7 == event.key:
            if isinstance(snd, Releasable):
                snd.release_at(None)
            logger.info("Rewind")
            return True  # reset
        # Arrows
        elif pg.K_UP == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                sampler.change_frametime(rel=step_size)
            else:
                # keyboard.move(-2, 0)
                keyboard.base *= 2.0
        elif pg.K_DOWN == event.key:
            if event.mod & (pg.KMOD_LALT | pg.KMOD_RALT):
                sampler.change_frametime(rel=-step_size)
            else:
                # keyboard.move(2, 0)
                keyboard.base /= 2.0
        elif pg.K_LEFT == event.key:
            keyboard.move(0, 1)
        elif pg.K_RIGHT == event.key:
            keyboard.move(0, -1)
        # Change frequency
        elif hasattr(snd, 'frequency'):
            change_frequency(snd, event.key)
            return True  # reset
    # Key up
    elif (event.type == pg.KEYUP and hasattr(snd, 'frequency')):
        if pg.K_CAPSLOCK == event.key:
            change_frequency(snd, event.key)
            return True  # reset
        else:
            if isinstance(snd, Releasable):
                try:
                    snd.release_at(watch.time())
                except TypeError:
                    logger.warning("Can't get current value from the iterator.")
                    snd.release_at(None)
                # logger.debug("Key '%s' (%s) up, released at: %s", pg.key.name(event.key), event.key, snd.released_at)
        return
    # Mouse
    elif hasattr(snd, 'frequency') and event.type in (
        pg.MOUSEBUTTONDOWN,
        pg.MOUSEBUTTONUP,
        pg.MOUSEMOTION
    ):
        if event.type == pg.MOUSEBUTTONDOWN:
            snd.pitch_bend = 0
        elif event.type == pg.MOUSEBUTTONUP:
            snd.pitch_bend = None
        elif (event.type == pg.MOUSEMOTION) and getattr(snd, 'pitch_bend', None) is not None:
            size = pg.display.get_surface().get_size()[1] or 0
            snd.pitch_bend = event.pos[1]

            logger.debug("Pitch bend == event.pos[0]: %s", snd.pitch_bend)

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
                "Pitched frequency to %s (with ratio %.04f) position %s, bend %s. [%s, %s, %s]",
                snd.frequency, ratio, event.pos[1], snd.pitch_bend,
                scale[0], scale[len(scale)//2], scale[-1]
            )
    else:
        logger.debug("Other: %s", event)
    return


def init_pygame(name="Resonance", size=800):
    """
    Initialize Pygame and return a surface.
    """
    pg.quit()

    logger.info(
        "Pygame initialized with %s loaded modules (%s failed)." % pg.init()
    )

    screen = init_display(name, size)
    logger.info("Inited display %s with flags: %s", screen, screen.get_flags())

    return screen


def init_display(name, size):
    """
    Initialize Pygame display and surface arrays.
    Returns Pygame screen.
    """
    pg.display.quit()

    flags = 0
    # flags |= pg.SRCALPHA
    flags |= pg.HWSURFACE
    # flags |= pg.OPENGL
    flags |= pg.DOUBLEBUF

    if 'numpy' in pg.surfarray.get_arraytypes():
        pg.surfarray.use_arraytype('numpy')
    else:
        raise ImportError('Numpy array package is not installed')

    # FIXME get resolution some other way.
    mode = pg.display.set_mode((size, size), flags, 32 if flags & pg.SRCALPHA else 24)
    pg.display.set_caption(name)
    pg.display.init()

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
        pg.mixer.get_init()
    )

    return pg.mixer.find_channel()


def cleanup():
    """
    Clean up: Quit pygame, close iterator.
    """
    logger.info("Doing cleanup.")
    pg.mixer.quit()
    pg.display.quit()
    pg.quit()


def change_frequency(snd, key):
    """
    Change frequency of the sound based on key position.
    """
    new_frequency = keyboard.get_frequency(key)
    if new_frequency == 0:
        return False
    snd.frequency = new_frequency
    if isinstance(snd, Releasable):
        snd.release_at(None)
    logger.debug("Changed frequency: %s.", snd.frequency if hasattr(snd, 'frequency') else snd)
    return new_frequency
