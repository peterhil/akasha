#!/usr/bin/env python
# -*- coding: utf-8 -*-

import exceptions
import logging
import numpy as np
import os
import pygame
import tempfile
import time

from PIL import Image

from akasha.funct import pairwise
from akasha.graphic.colour import hsv2rgb, angle2hsv, colorize, chords_to_hues, white
from akasha.graphic.primitive.line import line_linspace_cx
from akasha.timing import sampler
from akasha.utils.log import logger
from akasha.utils.math import clip, pad, pcm, complex_as_reals

try:
    import matplotlib.pyplot as plt
except:
    logger.warn("Can't import pyplot from matplolib!")
    pass


def hist_graph(samples, size=1000):
    """Uses numpy histogram2d to make an image from complex signal."""
    # NP Doc: http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    # TODO see also np.bincount and
    # http://stackoverflow.com/questions/7422713/numpy-histogram-with-3x3d-arrays-as-indices
    c = samples.view(np.float64).reshape(len(samples), 2).transpose()
    x, y = c[0], -c[1]
    hist, y_edges, x_edges = np.histogram2d(y, x, bins=size, range=[[-1.,1.],[-1.,1.]], normed=False)
    image = Image.fromarray(np.array(hist / hist.mean() * 255, dtype=np.uint8),'L')
    image.show()

def get_canvas(x_size=1000, y_size=None, axis=True):
    if not y_size:
        y_size = x_size
    img = np.zeros((y_size+1, x_size+1, 4), np.uint8)             # Note: y, x
    if axis:
        # Draw axis
        img[y_size/2.0,:] = img[:,x_size/2.0] = [42,42,42,127]
    return img

def get_points(samples, size=1000):
    # Scale to size and interpret values as pixel centers
    samples = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))      # 0.5 to 599.5
    return complex_as_reals(samples)

def draw(samples, size=1000, dur=None, antialias=False, lines=False, colours=True,
        axis=True, img=None, screen=None):
    """
    Draw the complex sound signal into specified size image.
    """
    # See http://jehiah.cz/archive/creating-images-with-numpy

    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner

    if img is not None: # Draw into existing img?
        size = img.shape[0]-1
    else:
        img = get_canvas(size, axis=axis)

    if dur:
        samples = samples[:int(round(dur * sampler.rate))]

    samples = clip_samples(samples)

    if lines:
        if antialias and screen is not None:
            draw_coloured_lines_aa(samples, screen, size, colours)
        else:
            # raise NotImplementedError("Drawing lines with Numpy is way too slow for now!")
            return draw_coloured_lines(samples, img, size, colours)
    else:
        if antialias:
            return draw_points_aa(samples, img, size, colours)
        else:
            return draw_points(samples, img, size, colours)

def clip_samples(samples):
    #clip_max = np.max(np.abs(samples)) # unit circle
    clip_max = np.max(np.fmax(np.abs(samples.real), np.abs(samples.imag))) # rectangular abs can be sqrt(2) > 1.0!
    if clip_max > 1.0:
        logger.warn("Clipping samples on draw() -- maximum magnitude was: %0.6f" % clip_max)
        return clip(samples)
    else:
        return samples

def add_alpha(rgb, alpha=255):
    return np.append(rgb, np.array([alpha] * len(rgb)).reshape(len(rgb), 1), 1)

def draw_coloured_lines_aa(samples, screen, size=1000, colours=True):
    """
    Draw antialiased lines with Pygame
    """
    if colours:
        # FIXME draws wrong colours on high frequencies!
        colors = colorize(samples)
        pts = get_points(samples, size).T
        for (i, ends) in enumerate(pairwise(pts)):
            #pts = get_points(np.array(ends), size).transpose()
            #color = hsv2rgb(angle2hsv(chords_to_hues(ends, padding=False)))
            #color = pygame.Color(*list(hsv2rgb(angle2hsv(chords_to_hues(ends, padding=False))))[:-1])
            pygame.draw.aaline(screen, colors[i], *ends)
    else:
        pts = get_points(samples, size).transpose()
        pygame.draw.aalines(screen, pygame.Color('orange'), False, pts, 1)

def draw_coloured_lines(samples, img, size=1000, colours=True):
    """
    Draw antialiased lines with Numpy
    """
    if len(samples) < 2:
        raise ValueError("Can't draw lines with less than two samples.")

    bg = get_canvas(size, axis=False)

    if colours:
        colors = add_alpha(colorize(samples))

    for i, (start, end) in enumerate(pairwise(pad(samples, -1))):
        line = np.linspace(start, end, np.abs(start - end) * size, endpoint=False)
        img += draw_points(line, bg, size, colours=False) * (colors[i] if colours else white)

    return img

def draw_points_np_aa(samples, img, size=1000, colours=True):
    """
    Draw colourized antialiased points from samples
    """
    points = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))
    deltas = points - np.round(points)

    color = add_alpha(colorize(samples)) if colours else white

    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5-0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5-0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5-0.5j)) * color
    img[:-1, :-1, :] = np.cast['int32'](complex_as_reals(deltas + -0.5-0.5j)) * color

    return img

def draw_points_aa(samples, img, size=1000, colours=True):
    """
    Draw colourized antialiased points from samples
    """
    points = get_points(samples, size)
    centers = np.round(points)  # 1.0 to 600.0
    bases = np.cast['int32'](centers) - 1  # 0 to 599
    deltas = points - bases - 0.5

    values_00 = deltas[1] * deltas[0]
    values_01 = deltas[1] * (1.0 - deltas[0])
    values_10 = (1.0 - deltas[1]) * deltas[0]
    values_11 = (1.0 - deltas[1]) * (1.0 - deltas[0])

    pos = [
        ( (size-1) - bases[1], bases[0] ),
        ( (size-1) - bases[1], bases[0]+1 ),
        ( (size) - (bases[1]+1), bases[0] ),
        ( (size) - (bases[1]+1), bases[0]+1 ),
    ]

    color = add_alpha(colorize(samples)) if colours else white

    img[pos[0][1], pos[0][0], :] += color * np.repeat(values_11, 4).reshape(len(samples), 4)
    img[pos[1][1], pos[1][0], :] += color * np.repeat(values_10, 4).reshape(len(samples), 4)
    img[pos[2][1], pos[2][0], :] += color * np.repeat(values_01, 4).reshape(len(samples), 4)
    img[pos[3][1], pos[3][0], :] += color * np.repeat(values_00, 4).reshape(len(samples), 4)
    return img

def draw_points(samples, img, size=1000, colours=True):
    """
    Draw colourized points from samples
    """
    points = get_points(samples, size)
    points = np.cast['uint32'](points)  # 0 to 599

    color = add_alpha(colorize(samples)) if colours else white
    img[points[0], (size - 1) - points[1]] = color
    return img


def video_transfer(signal, type='PAL', axis='real', horiz=720):
    # See http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    # Stereo?
    #
    # PAL:
    # 294 × 50 × 3 = 44,100
    # 294 active lines/field × 50 fields/second × 3 samples/line = 44,100 samples/second
    # (588 active lines per frame, out of 625 lines total)

    # NTSC:
    # 245 × 60 × 3 = 44,100
    # 245 active lines/field × 60 fields/second × 3 samples/line = 44,100 samples/second
    # (490 active lines per frame, out of 525 lines total)

    # See 576i and 576p: horiz. 720 or 704, vert. 576 out of 625 lines
    formats = { 'PAL': 588, 'NTSC': 490 }
    vert = formats[type]

    linewidth = 3   # samples per line
    framesize = vert * linewidth   # 1764 for PAL, 1470 for NTSC

    # Make complex signal real
    if isinstance(signal[0], np.complex):
        signal = getattr(signal, axis)

    #for block in xrange(0, len(signal), framesize):
    #    pass # draw frame

    img = get_canvas(3 - 1, vert - 1, axis=False) # Stretch to horiz. width later!
    fv = img.flat

    s = pcm(signal[:framesize] * 256, bits=8, axis='real').astype(np.uint8)
    #fv[::] = np.repeat(signal[:framesize].T, 4, axis=0).T  # y, x
    fv[::] = np.repeat(s, 4, axis=0)

    #logger.debug("Image:\n%s,\nFlat view:\n%s" % (img[:framesize], fv[:framesize]))
    return np.repeat(img[:framesize], horiz / linewidth, axis=1)


# Showing images

def show(img, plot=False, osx_open=False):
    if (plot and plt):
        plt.interactive(True)
        imgplot = plt.imshow(img[:,:,:3])
        imgplot.set_cmap('hot')
        plt.show(False)
    elif osx_open:
        try:
            tmp = tempfile.NamedTemporaryFile(dir='/var/tmp', suffix='akasha.bmp')
            logger.debug("Tempfile: %s" % tmp.name)
            image.save(tmp, 'bmp')
            time.sleep(0.5)
            os.system("open " + tmp.name)
        except IOError, err:
            logger.error("Failed to open a temporary file and save the image: %s" % err)
        except OSError, err:
            logger.error("Failed to open the image with a default app: %s" % err)
        finally:
            tmp.close()
    else:
        image = Image.fromarray(img[...,:3], 'RGB')
        image.show()


def fast_graph(samples, size=1000, plot=False):
    return graph(samples, size, plot, antialias=False)

def graph(samples, size=1000, dur=None, plot=False, axis=True,
        antialias=True, lines=False, colours=True, img=None):
    if dur:
        samples = samples[:int(round(dur * sampler.rate))]
    img = draw(samples, size=size,
            antialias=antialias, lines=lines, colours=colours,
            axis=axis, img=img).transpose((1, 0, 2))
    show(img, plot)
    return False

def plot(samples):
    "Plot samples using matplotlib"
    imgplot = plt.imshow([samples.real, samples.imag])
    imgplot.set_cmap('hot')
    return False

def plot_real_fn(fn, x, cmap='hot'):
    plt.set_cmap(cmap)
    y = fn(x)
    plt.plot(x, y)
    return False

