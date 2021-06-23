#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PySDL2 GUI module
"""

import numpy as np
import pygame as pg
import sdl2
import sdl2.ext

from akasha.math import pcm
from akasha.timing import sampler
from akasha.utils import issequence
from akasha.utils.log import logger


class PySDL2Gui:
    def __init__(self):
        self.clock = pg.time.Clock()

    def init(self, name="Resonance", size=800):
        """
        Initialize PySDL2 and return a surface.
        """
        sdl2.SDL_Quit()

        logger.info(
            "PySDL2 initialized: %s" % sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
        )

        self.window = window = self.init_window(name, size)
        logger.info("Initialised window %s", window)
        self.screen = sdl2.SDL_GetWindowSurface(window)

        return window

    def init_window(self, name, size):
        """
        Initialize PySDL2 display and surface arrays.
        Returns PySDL2 screen.
        """
        sdl2.ext.quit()

        window = sdl2.SDL_CreateWindow(
            name,
            sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
            size, size,
            sdl2.SDL_WINDOW_SHOWN
        )

        return window

    def init_mixer(self, *args):
        """
        Initialize the PySDL2 mixer.
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

    def blit(self, img):
        """
        Blit the screen.
        """
        def to_rgb_pixel(rgb):
            return rgb[0] << 16 + rgb[1] << 8 + rgb[2]
        if self.screen and img is not None:  # Pygame drawing methods do not return img
            array_surface = sdl2.ext.pixels3d(self.screen.contents)
            # data = np.apply_along_axis(to_rgb_pixel, 2, img[..., :3])  # Drop alpha
            # data = img[..., 0].flatten().reshape(800, 800)  # Drop alpha
            # np.copyto(array_surface, data)
            array_surface[..., :3] = img[..., :3]
            # sdl2.SDL_LockSurface(array_surface)
            # img_surface = sdl2.SDL_CreateRGBSurfaceFrom(array_surface.pixels, 800, 800, 24, 800, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x0)
            # sdl2.SDL_UnlockSurface(array_surface)
            sdl2.SDL_BlitSurface(array_surface, None, self.screen, None)
            # self.window.refresh()

    def cleanup(self):
        """
        Clean up: Quit pygame, close iterator.
        """
        logger.info("Doing cleanup.")
        pg.mixer.quit()
        sdl2.ext.quit()
        sdl2.SDL_Quit()

    def tick(self, rate):
        return self.clock.tick_busy_loop(rate)

    def flip(self):
        # return sdl2.ext.Window.refresh(self.window)
        return True

    @staticmethod
    def get_size():
        return pg.display.get_surface().get_size()

    @staticmethod
    def get_events():
        return sdl2.ext.get_events()

    @staticmethod
    def queue_audio(samples, channel):
        """
        Queue samples into a mixer channel.
        """
        return channel.queue(pg.sndarray.make_sound(pcm(samples)))

    @staticmethod
    def get_key(event):
        return event.key.keysym.sym

    @staticmethod
    def key_pause(event):
        return (event.type == sdl2.SDL_KEYDOWN and event.key == sdl2.SDLK_F8) \
          or (event.type == sdl2.SDL_WINDOWEVENT and event.window.event in [
              sdl2.SDL_WINDOWEVENT_FOCUS_GAINED,
              sdl2.SDL_WINDOWEVENT_FOCUS_LOST])

    @staticmethod
    def key_escape(event):
        return event.type == sdl2.SDL_QUIT or (event.type == sdl2.SDL_KEYDOWN and event.key == sdl2.SDLK_ESCAPE)

    @staticmethod
    def keydown(event):
        return event.type == sdl2.SDL_KEYDOWN

    @staticmethod
    def keyup(event):
        return event.type == sdl2.SDL_KEYUP

    @staticmethod
    def keyname(event):
        return sdl2.SDL_GetKeyName(event.key.keysym.sym)

    @staticmethod
    def key_alt(event):
        return sdl2.SDL_GetModState() & sdl2.KMOD_ALT

    @staticmethod
    def key_shift(event):
        return sdl2.SDL_GetModState() & sdl2.KMOD_SHIFT

    @staticmethod
    def key_f7(event):
        return sdl2.SDLK_F7 == event.key

    @staticmethod
    def key_up(event):
        return sdl2.SDLK_UP == event.key

    @staticmethod
    def key_down(event):
        return sdl2.SDLK_DOWN == event.key


    @staticmethod
    def key_left(event):
        return sdl2.SDLK_LEFT == event.key

    @staticmethod
    def key_right(event):
        return sdl2.SDLK_RIGHT == event.key

    @staticmethod
    def key_caps_lock(event):
        return sdl2.SDLK_CAPSLOCK == event.key

    @staticmethod
    def mouse_event(event):
        return event.type in (
            pg.MOUSEBUTTONDOWN,
            pg.MOUSEBUTTONUP,
            pg.MOUSEMOTION
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
