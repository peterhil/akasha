#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pygame GUI module
"""

import pygame as pg

from akasha.math import pcm
from akasha.settings import config
from akasha.timing import sampler
from akasha.utils.array import is_sequence
from akasha.utils.log import logger


class PygameGui:
    def __init__(self):
        self.clock = pg.time.Clock()

    def init(self, name="Resonance", size=800):
        """
        Initialize Pygame and return a surface.
        """
        pg.quit()
        loaded, failed = pg.init()
        logger.info(
            'Pygame initialized with %s loaded modules ' '(%s failed).',
            loaded,
            failed,
        )

        screen = self.init_display(name, size)
        logger.info(
            'Inited display %s with flags: %s',
            screen,
            screen.get_flags(),
        )

        return screen

    def init_display(self, name, size):
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

        bitdepth = 32 if flags & pg.SRCALPHA else 24
        mode = pg.display.set_mode((size, size), flags, bitdepth)
        pg.display.set_caption(name)
        pg.display.init()

        return mode

    def init_mixer(self, *args):
        """
        Initialize the Pygame mixer.
        """
        pg.mixer.quit()

        # Set mixer defaults: sample rate, sample size,
        # number of channels, buffer size
        if is_sequence(args) and 0 < len(args) <= 3:
            pg.mixer.init(*args)
        else:
            pg.mixer.init(
                frequency=sampler.rate,
                size=config.audio.SAMPLETYPE,
                channels=config.audio.CHANNELS,
                buffer=config.audio.BUFFERSIZE,
            )

        fs, size, channels = pg.mixer.get_init()
        logger.info(
            "Mixer has %s Hz sample rate with %s size samples and "
            "%s channels.",
            fs,
            size,
            channels,
        )

        return pg.mixer.find_channel(force=True)

    def cleanup(self):
        """
        Clean up: Quit pygame, close iterator.
        """
        logger.info("Doing cleanup.")
        pg.mixer.quit()
        pg.display.quit()
        pg.quit()

    def tick(self, rate):
        return self.clock.tick_busy_loop(rate)

    @staticmethod
    def flip():
        return pg.display.flip()

    @staticmethod
    def get_size():
        return pg.display.get_surface().get_size()

    @staticmethod
    def get_event():
        return pg.event.get()

    @staticmethod
    def queue_audio(samples, channel):
        """
        Queue samples into a mixer channel.
        """
        waveform = pcm(samples, bits=config.audio.SAMPLETYPE)
        return channel.queue(pg.sndarray.make_sound(waveform))

    @staticmethod
    def key_pause(event):
        return (event.type == pg.KEYDOWN and event.key == pg.K_F8) or (
            event.type == pg.ACTIVEEVENT and event.state == 3
        )

    @staticmethod
    def key_escape(event):
        return event.type == pg.QUIT or (
            event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
        )

    @staticmethod
    def keydown(event):
        return event.type == pg.KEYDOWN

    @staticmethod
    def keyup(event):
        return event.type == pg.KEYUP

    @staticmethod
    def keyname(event):
        return pg.key.name(event.key)

    @staticmethod
    def key_alt(event):
        return event.mod & (pg.KMOD_LALT | pg.KMOD_RALT)

    @staticmethod
    def key_shift(event):
        return event.mod & (pg.KMOD_LSHIFT | pg.KMOD_RSHIFT)

    @staticmethod
    def key_f7(event):
        return pg.K_F7 == event.key

    @staticmethod
    def key_up(event):
        return pg.K_UP == event.key

    @staticmethod
    def key_down(event):
        return pg.K_DOWN == event.key

    @staticmethod
    def key_left(event):
        return pg.K_LEFT == event.key

    @staticmethod
    def key_right(event):
        return pg.K_RIGHT == event.key

    @staticmethod
    def key_caps_lock(event):
        return pg.K_CAPSLOCK == event.key

    @staticmethod
    def mouse_event(event):
        return event.type in (
            pg.MOUSEBUTTONDOWN,
            pg.MOUSEBUTTONUP,
            pg.MOUSEMOTION,
        )

    @staticmethod
    def mouse_down(event):
        return event.type == pg.MOUSEBUTTONDOWN

    @staticmethod
    def mouse_up(event):
        return event.type == pg.MOUSEBUTTONUP

    @staticmethod
    def mouse_motion(event):
        return event.type == pg.MOUSEMOTION
