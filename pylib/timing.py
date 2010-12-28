#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from timeit import default_timer as clock


class Sampler(object):
    rate = 44100
    videorate = 30

# @staticmethod
def times_at(frames):
    """Convert frame numbers to time.

    >>> time_at(44100)
    1.0
    """
    return frames / float(Sampler.rate)

# @staticmethod
def frames_at(times):
    """Convert time to frame numbers (ie. 1.0 => 44100)"""
    return int(round(times * Sampler.rate))

class Timeline(object):
    pass

class OutputStream(object):
    pass
