#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
App with graphical user interface (GUI) done with Pygame
"""

import pygame as pg


class Unity(object):
    """
    Unity is the original Pygame animated signal that can be played with keyboard and mouse.
    """
    def __init__(self, width, height=None, flags=pg.HWSURFACE | pg.DOUBLEBUF):
        self._running = False
        self._surface = None
        self.size = width, height or width
        self.flags = int(flags)

    def init(self):
        pg.init()
        self._surface = pg.display.set_mode(self.size, self.flags)
        self._running = True

    def cleanup(self):
        pg.quit()

    def event(self, event):
        if event.type == pg.QUIT:
            self._running = False

    def update(self):
        pass

    def render(self):
        pass

    def execute(self):
        if self.init() == False:
            self._running = False

        while self._running:
            for event in pg.event.get():
                self.event(event)
            self.update()
            self.render()

        self.cleanup()


if __name__ == "__main__" :
    app = Unity(800)
    app.execute()
