#!/usr/bin/env python

from timeit import default_timer as timer


class Timed:
    """
    Time some code using with statement.
    """

    elapsed = 0

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.end = timer()
        self.elapsed = self.end - self.start

    def __float__(self):
        return float(self.elapsed)
