#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Keyboard and mouse input using curses
"""


import curses
import locale

locale.setlocale(locale.LC_ALL, '')
code = locale.getlocale()[1]

def func(self):
    stdscr = curses.initscr()
    print code
    message = u"hello わたし!"
    stdscr.addstr(0, 0, message.encode("utf-8"), curses.A_BLINK)
    # curses.echo()
    while 1:
        c = stdscr.getch()
        # if c == ord('q'): break  # Exit the while()
        if c == 27: break   # escape, Exit the while()
        else:
            stdscr.addstr(0, 0, unicode(c) + u' ' + unichr(c).encode('utf-8'))

if __name__=='__main__':
    curses.wrapper(func)