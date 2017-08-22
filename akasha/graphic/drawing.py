#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Graphics drawing module.
"""

import numpy as np
import os
import pygame
import tempfile
import time

from akasha.curves.circle import Circle
from akasha.funct import pairwise
from akasha.graphic.colour import colorize, white
# from akasha.graphic.colour import hsv2rgb, angle2hsv, chords_to_hues
from akasha.graphic.primitive.line import bresenham
from akasha.timing import sampler
from akasha.utils.log import logger
from akasha.math import \
    clip, \
    complex_as_reals, \
    flip_vertical, \
    get_pixels, \
    get_points, \
    inside, \
    normalize, \
    pad, \
    pcm

from itertools import izip
from PIL import Image
from scipy import sparse
from skimage import draw as skdraw

try:
    import matplotlib.pyplot as plt
except ImportError:
    logger.warn("Can't import pyplot from matplolib!")
    plt = None

# TODO: Use scipy.sparse matrices to speed up things?
# Use sparse.coo (coordinate matrix) to build the matrix, and convert to csc/csr for math.


def draw_axis(img, colour=[42, 42, 42, 127]):
    """
    Draw axis on the image with the colour.
    """
    height, width, channels = img.shape
    img[height / 2.0, :] = img[:, width / 2.0] = colour[:channels]

    return img


def get_canvas(width=1000, height=None, channels=4, axis=True):
    """
    Get a Numpy array suitable for use as a drawing canvas.
    """
    if height is None:
        height = width

    img = np.zeros((height, width, channels), np.uint8)  # Note: y, x

    # FIXME: axis argument for get_canvas should accept a colour value, or it shouldn't exist
    if axis:
        img = draw_axis(img)

    return img


def blit(screen, img):
    """
    Blit the screen.
    """
    if screen and img is not None:
        pygame.surfarray.blit_array(screen, img[..., :3])  # Drop alpha


def draw(
    signal,
    size=1000,
    antialias=False,
    lines=False,
    colours=True,
    axis=True,
    img=None,
    screen=None,
):
    """
    Draw the complex sound signal into specified size image.
    """
    # See: http://jehiah.cz/archive/creating-images-with-numpy

    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # signal = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner

    # TODO: Enable using non-square size.

    if img is not None:  # Draw into existing img?
        size = img.shape[0]
    else:
        img = get_canvas(size, axis=axis)

    if len(signal) == 0:
        logger.warn('Drawing empty signal!')
        return img

    signal = clip_samples(signal)

    if lines:
        if screen is None:
            # logger.warn("Drawing lines with Numpy is way too slow for now!")
            return draw_lines(signal, img, size, colours)
        if antialias and screen is not None:
            draw_lines_pg(signal, screen, size, colours, antialias=True)
        else:
            draw_lines_pg(signal, screen, size, colours, antialias=False)
    else:
        if antialias:
            return draw_points_aa(signal, img, size, colours)
        else:
            return draw_points(signal, img, size, colours)


def clip_samples(signal):
    """
    Clip a signal into unit rectangle area.
    """
    # TODO: Move to math or dsp module

    # clip_max = np.max(np.abs(signal)) # unit circle
    # rectangular abs can be sqrt(2) > 1.0!
    signal = np.atleast_1d(signal)
    clip_max = np.max(np.fmax(np.abs(signal.real), np.abs(signal.imag)))
    if clip_max > 1.0:
        logger.warn("Clipping signal -- maximum magnitude was: %0.6f" % clip_max)
        return clip(signal)
    else:
        return signal


def add_alpha(rgb, opacity=255):
    """
    Add alpha channel with specified opacity to the rgb signal.
    """
    return np.append(rgb, np.array([opacity] * len(rgb)).reshape(len(rgb), 1), 1)


def draw_lines_pg(signal, screen, size=1000, colours=True, antialias=False):
    """
    Draw (antialiased) lines with Pygame.
    """
    img = get_canvas(size, axis=True)
    blit(screen, img)

    method = 'aaline' if antialias else 'line'
    pts = get_points(flip_vertical(signal), size).T
    if colours:
        # FIXME: aaline draws wrong colours on high frequencies!
        colors = add_alpha(colorize(signal))
        for (i, ends) in enumerate(pairwise(pts)):
            getattr(pygame.draw, method)(screen, colors[i], *ends)
    else:
        getattr(pygame.draw, method + 's')(screen, pygame.Color('orange'), False, pts, 1)


def draw_lines(signal, img, size=1000, colours=True):
    """
    Draw antialiased lines with Numpy.
    """
    if len(signal) < 2:
        signal = pad(signal, -1)
    if colours:
        colors = add_alpha(colorize(signal))

    points = np.rint(get_points(flip_vertical(signal), size) - 0.5).astype(np.uint32).T
    segments = np.hstack((points[:-1], points[1:]))

    for i, coords in enumerate(segments):
        img[skdraw.bresenham(*coords)] = colors[i] if colours else white

    return img


def draw_points_np_aa(signal, img, size=1000, colours=True):
    """
    Draw colourized antialiased points from signal.
    """
    points = ((clip(signal) + 1 + 1j) / 2.0 * (size - 1) + (0.5 + 0.5j))
    deltas = points - np.round(points)

    color = add_alpha(colorize(signal)) if colours else white

    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5 - 0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5 - 0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5 - 0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5 - 0.5j)) * color

    return img


def draw_points_aa(signal, img, size=1000, colours=True):
    """
    Draw colourized antialiased points from signal.
    """
    # Fixme: Ignores size argument as it is now

    width, height, ch = img.shape

    iw = lambda a: inside(a, 0, width)
    ih = lambda a: inside(a, 0, height)
    area = lambda w: np.repeat(w, ch).reshape(len(signal), ch)

    px, value = get_pixels(signal, width - 1)
    color = add_alpha(colorize(signal)) if colours else white

    img[px[0],         px[1], :]         += color * area(value[0] * value[1])
    img[px[0],         iw(px[1] + 1), :] += color * area(value[0] * (1 - value[1]))
    img[ih(px[0] + 1), px[1], :]         += color * area((1 - value[0]) * value[1])
    img[ih(px[0] + 1), iw(px[1] + 1), :] += color * area((1 - value[0]) * (1 - value[1]))

    return img


def draw_points_aa_old(signal, img, size=1000, colours=True):
    """
    Draw colourized antialiased points from signal.
    """
    size -= 1
    bases, deltas = get_pixels(signal, size)

    values_00 = deltas[1] * deltas[0]
    values_01 = deltas[1] * (1.0 - deltas[0])
    values_10 = (1.0 - deltas[1]) * deltas[0]
    values_11 = (1.0 - deltas[1]) * (1.0 - deltas[0])

    pos = [
        ((size - 1) - bases[1], bases[0]),
        ((size - 1) - bases[1], bases[0] + 1),
        ((size - 1) - (bases[1] + 1), bases[0]),
        ((size - 1) - (bases[1] + 1), bases[0] + 1),
    ]

    color = add_alpha(colorize(signal)) if colours else white

    img[pos[0][1], pos[0][0], :] += color * np.repeat(values_11, 4).reshape(len(signal), 4)
    img[pos[1][1], pos[1][0], :] += color * np.repeat(values_10, 4).reshape(len(signal), 4)
    img[pos[2][1], pos[2][0], :] += color * np.repeat(values_01, 4).reshape(len(signal), 4)
    img[pos[3][1], pos[3][0], :] += color * np.repeat(values_00, 4).reshape(len(signal), 4)

    return img


def draw_points(signal, img, size=1000, colours=True):
    """
    Draw colourized points from signal
    """
    points = np.rint(get_points(flip_vertical(signal), size) - 0.5).astype(np.uint32)

    color = add_alpha(colorize(signal)) if colours else white

    img[points[0], points[1]] = color[..., :img.shape[2]]
    return img


def draw_points_coo(signal, img, size=1000, colours=True):
    """
    Draw a bitmap image from a complex signal with optionally colourized pixels.
    """
    points = np.rint(get_points(flip_vertical(signal), size) - 0.5).astype(np.uint32)
    coords = sparse.coo_matrix(points, dtype=np.uint32).todense()

    color = add_alpha(colorize(signal)) if colours else white

    img[coords[0], coords[1]] = color[..., :img.shape[2]]
    return img


def hist_graph(signal, size=1000):
    """
    Use Numpy histogram2d to make an image from the complex signal.
    """
    # NP Doc: http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    #
    # TODO: see also np.bincount and Numpy histogram with 3x3d arrays as indices
    # http://stackoverflow.com/questions/7422713/numpy-histogram-with-3x3d-arrays-as-indices
    x, y = get_points(flip_vertical(signal))
    histogram, _, _ = np.histogram2d(
        y,
        x,
        bins=size,
        range=[[-1., 1.], [-1., 1.]],
        normed=True
    )
    histogram = normalize(histogram)
    image = Image.fromarray(np.array(histogram * 255, dtype=np.uint8), 'L')
    image.show()
    return histogram


def video_transfer(signal, standard='PAL', axis='real', horiz=720):
    """
    Draw a sound signal using the old video tape audio recording technique.
    See: http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """
    # TODO: Make Stereo video transfer?

    # PAL:
    # 294 × 50 × 3 = 44,100
    # 294 active lines/field × 50 fields/second × 3 samples/line = 44,100 samples/second
    # (588 active lines per frame, out of 625 lines total)

    # NTSC:
    # 245 × 60 × 3 = 44,100
    # 245 active lines/field × 60 fields/second × 3 samples/line = 44,100 samples/second
    # (490 active lines per frame, out of 525 lines total)

    # See 576i and 576p: horiz. 720 or 704, vert. 576 out of 625 lines
    formats = {
        'PAL': 588,
        'NTSC': 490,
    }
    vert = formats[standard]

    linewidth = 3   # samples per line
    framesize = vert * linewidth   # 1764 for PAL, 1470 for NTSC

    # Make complex signal real
    if isinstance(signal[0], np.complex):
        signal = getattr(signal, axis)

    #for block in xrange(0, len(signal), framesize):
    #    pass # draw frame

    img = get_canvas(3, vert, axis=False)  # Stretch to horiz. width later!
    fv = img.flat

    s = pcm(signal[:framesize] * 256, bits=8, axis='real').astype(np.uint8)
    #fv[::] = np.repeat(signal[:framesize].T, 4, axis=0).T  # y, x
    fv[::] = np.repeat(s, 4, axis=0)

    #logger.debug("Image:\n%s,\nFlat view:\n%s" % (img[:framesize], fv[:framesize]))
    return np.repeat(img[:framesize], horiz / linewidth + 2, axis=1)


# Showing images

def show(img, plot=False, osx_open=False):
    """
    Show an image from a Numpy array.
    """
    img = img.transpose((1, 0, 2))
    if (plot and plt):
        plt.interactive(True)
        imgplot = plt.imshow(img[:, :, :3])
        imgplot.set_cmap('hot')
        plt.show(False)
    elif osx_open:
        try:
            tmp = tempfile.NamedTemporaryFile(dir='/var/tmp', suffix='akasha.png')
            logger.debug("Tempfile: %s" % tmp.name)
            image = Image.fromarray(img[..., :3], 'RGB')
            image.save(tmp, 'png')
            time.sleep(0.5)
            os.system("open " + tmp.name)
        except IOError, err:
            logger.error("Failed to open a temporary file and save the image: %s" % err)
        except OSError, err:
            logger.error("Failed to open the image with a default app: %s" % err)
        finally:
            tmp.close()
    else:
        image = Image.fromarray(img[..., :3], 'RGB')
        image.show()


def imsave(img, filename):
    try:
        img = img.transpose((1, 0, 2))
        image = Image.fromarray(img[..., :3], 'RGB')
        image.save(filename, 'png')
    except IOError, err:
        logger.error("Failed to save image into {}".format(filename))
    except Exception, err:
        logger.error("Error when saving image: {}".format(err))


def graph(signal, size=1000, dur=None, plot=False, axis=True,
          antialias=True, lines=False, colours=True, img=None):
    """
    Make an image from the sound signal and show it.
    """
    if dur:
        signal = signal[:int(round(dur * sampler.rate))]

    img = draw(
        signal,
        size=size,
        antialias=antialias, lines=lines, colours=colours,
        axis=axis,
        img=img
    )

    show(img, plot and plt)
