#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Graphics plotting module.
"""

import numpy as np
import pylab

from contextlib import contextmanager

from akasha.curves.circle import Circle
from akasha.math import complex_as_reals


@contextmanager
def plotting():
    """
    Context manager for plotting an image interactively using pylab.
    """
    pylab.interactive(True)
    try:
        yield
    finally:
        pylab.axis('equal')
        pylab.show(block=False)


def plot_complex_image(signal, cmap='hot'):
    """
    Show complex signal image using pylab.imshow()
    """
    with plotting():
        im = pylab.imshow([signal.real, signal.imag], cmap)
    return im


def plot_signal(signal):
    """
    Plot complex signal using pylab.plot()
    """
    with plotting():
        fig = pylab.plot(*complex_as_reals(signal))
    return fig


def plot_real_fn(fn, x):
    """
    Plot a real valued function with x values using pylab.plot()
    """
    with plotting():
        fig = pylab.plot(x, fn(x))
    return fig


def plot_unit(axes=3, scale=1, n=800):
    """
    Helper function to plot an unit circle
    """
    samples = np.linspace(0, 1, n, endpoint=False)
    o = Circle.at(samples) * scale

    with plotting():
        fig = pylab.plot(o.real, o.imag)
        pylab.axis((-axes, axes, -axes, axes))
    return fig
