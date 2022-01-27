#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from timeit import default_timer as timer

from akasha.utils.array import is_empty


class Watch():
    def __init__(self, maxstops=5):
        self.paused = 0
        self.maxstops = maxstops
        self.timings = []
        self.reset()

    def reset(self):
        self.epoch = timer()
        if self.paused:
            self.paused = self.epoch
        self.lasttime = 0

    def time(self):
        if not self.paused:
            return timer() - self.epoch
        else:
            return self.paused - self.epoch

    def last(self):
        return self.time() - self.lasttime

    def next(self):
        if not self.paused:
            self.lasttime = self.time()
            self.timings.append(self.lasttime)
            self.timings = self.timings[-self.maxstops:]
        return self.lasttime

    def pause(self):
        if not self.paused:
            self.paused = timer()
        else:
            if self.paused > 0:
                self.epoch += timer() - self.paused
            self.paused = 0

    def get_fps(self, n=None):
        if n is None:
            ts = np.ediff1d(np.array(self.timings))
        elif n > 0:
            ts = np.ediff1d(np.array(self.timings[-n:]))

        # print('timings:', ts)
        ts = ts[ts >= 2e-3]  # Filter trash values after reset
        if is_empty(ts): return 0

        return np.median(1.0 / ts)
