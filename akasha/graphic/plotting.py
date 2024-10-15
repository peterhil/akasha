"""
Graphics plotting module.
"""

import numpy as np
import pylab as lab

from contextlib import contextmanager

from akasha.curves.circle import Circle
from akasha.math import complex_as_reals


@contextmanager
def plotting():
    """
    Context manager for plotting an image interactively using pylab.
    """
    lab.interactive(True)
    try:
        yield
    finally:
        lab.axis('equal')
        lab.show(block=False)


def plot_complex_image(signal, cmap='hot'):
    """
    Show complex signal image using pylab.imshow()
    """
    with plotting():
        im = lab.imshow([signal.real, signal.imag], cmap)
    return im


def plot_signal(signal):
    """
    Plot complex signal using pylab.plot()
    """
    with plotting():
        fig = lab.plot(*complex_as_reals(signal))
    return fig


def plot_real_fn(fn, x):
    """
    Plot a real valued function with x values using pylab.plot()
    """
    with plotting():
        fig = lab.plot(x, fn(x))
    return fig


def plot_unit(axes=3, scale=1, n=800):
    """
    Helper function to plot an unit circle
    """
    samples = np.linspace(0, 1, n, endpoint=False)
    o = Circle.at(samples) * scale

    with plotting():
        fig = lab.plot(o.real, o.imag)
        lab.axis((-axes, axes, -axes, axes))
    return fig
