#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Periodic (and tesselation) arrays.
"""

import inspect
import numpy as np

from numbers import Number

from akasha.utils import _super

debug = False
debug_gs = False


def whoami():
    """
    What's my name?
    """
    return inspect.stack()[1][3]


class period(np.ndarray, object):
    """
    Periodic n-dimensional array. Subclasses numpy.ndarray.

    The periodicity in each dimension is determined by the array shape.
    Useful for signal processing, where the periodic functions (sin, cos, exp) are often used.

    Periodic array is also a kind of circular buffer, but differs form the usual implementation
    in that all access is made modulo some period(s) in the dimension(s) in question.

    Examples
    ========

    @todo

    """
    def __new__(cls, *args, **kwargs):
        if debug:
            print('In __new__ with class %s' % cls)
        return np.ndarray.__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj):
        if debug:
            print('In array_finalize:')
            print('   self type is %s' % type(self))
            print('   obj type is %s' % type(obj))

    @classmethod
    def array(cls, seq, *args, **kwargs):
        """
        Create a periodic array from a sequence. Accepts the same arguments as numpy.array().
        """
        seq = np.array(seq)

        if not hasattr(kwargs, 'dtype'):
            kwargs['dtype'] = seq.dtype

        out = cls.__new__(cls, seq.shape, *args, **kwargs)

        if seq.ndim > 0:
            out[::] = seq

        return out

    def _mod(self, index, dim=None):
        """Modulate indices to self.shape."""
        if debug_gs:
            print("Type in _mod: %s %s" % (index, type(index)))

        # if np.isscalar(index):
        if isinstance(index, Number):
            return np.mod(index, self.shape[dim])  # pylint: disable=E1101
        elif isinstance(index, slice):
            return self._mod_slice(index, dim)
        elif isinstance(index, np.ndarray):
            return self._mod_seq(index, dim)
        elif isinstance(index, (tuple, list)):  # sequence
            if debug_gs:
                print("Enumerating mixed index:")

            out = np.array(index)
            for i, item in enumerate(index):
                if debug_gs:
                    print(i, item)

                if np.isscalar(item):
                    out[i] = np.mod(item, self.shape[dim])  # pylint: disable=E1101
                else:
                    out[i] = self._mod(item, dim)
            return out
        else:
            raise ValueError(
                "Expected 'index' to be either an int, slice or sequence, "
                "or a tuple of the previous items, got: %s" %
                type(index)
            )

    def _mod_seq(self, seq, dim=None):
        """Modulate sequence to self.shape."""
        if np.isscalar(seq):
            raise ValueError("Expected a sequence, got: %s %s" % (seq, type(seq)))
        index = np.array(seq, dtype=np.int64)
        return np.mod(index, self.shape[dim])  # pylint: disable=E1101

    def _mod_slice(self, sl, dim=None):
        """
        Normalise slicing mod self.shape. If given a slice the behaviour will be:

        - Step defaults to 1, is wrapped modulo period, and can't be zero!
        - Start defaults to 0, is wrapped modulo period
        - Number of elements returned is the absolute differerence of
        - stop - start (or period and 0 if either value is missing)
        - Element count is multiplied with step to produce the same
        - number of elements for different step values.
        """
        # pylint: disable=E1101

        # if isinstance(sl, slice):
        #     return slice(
        #         sl.start and (sl.start % self.shape[dim]) or 0,
        #         sl.stop  and (sl.stop  % self.shape[dim]) or self.shape[dim],
        #         sl.step  and (sl.step  % self.shape[dim]) or 1,
        #     )
        if isinstance(sl, slice):
            step = ((sl.step or 1) % self.shape[dim] or 1)
            start = ((sl.start or 0) % self.shape[dim])
            count = abs((sl.stop or self.shape[dim]) - (sl.start or 0))
            stop = start + (count * step)
            return slice(start, stop, step)
            # return np.arange(*(slice(start, stop, step).indices(stop)))
        else:
            raise ValueError("Expected a slice, got: %s" % type(sl))

    def __getitem__(self, index):
        if debug_gs:
            print("\nIn %s %s: i=%s" % (__name__, whoami(), index))

        out = self._mod(index, 0)

        if debug_gs:
            print("Modded to: %s %s" % (out, type(out)))

        return _super(self).__getitem__(
            out
        )

    def __setitem__(self, index, value):
        if debug_gs:
            print("In %s %s: i=%s y=%s" % (__name__, whoami(), index, value))

        _super(self).__setitem__(
            self._mod(index, 0),
            value
        )

    # def __getslice__(self, i, j):
    #     if debug_gs: print "In %s %s: i=%s j=%s" % (__name__, whoami(), i, j)
    #     return _super(self).__getslice__(
    #         int(i), #self._mod(i, 0),
    #         int(j), #self._mod(j, 1)
    #     )

    # def __setslice__(self, i, j, y):
    #     if debug_gs: print "In %s %s: i=%s j=%s y=%s" % (__name__, whoami(), i, j, y)
    #     _super(self).__setslice__(
    #         i, #self._mod(i, 0),
    #         j, #self._mod(j, 1),
    #         y
    #     )
