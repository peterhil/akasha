#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import colorsys
import exceptions
import logging
import numpy as np
import pygame

from cmath import phase
from PIL import Image

from .decorators import memoized
from .math import normalize, clip, deg, distances, pad, pcm, minfloat, complex_as_reals
from .log import logger

from ..funct import pairwise
from ..timing import Sampler

try:
    import matplotlib.pyplot as plt
except:
    logger.warn("Can't import pyplot from matplolib!")
    pass


lowest_audible_hz = 16.35

# Colour conversions from: http://local.wasp.uwa.edu.au/~pbourke/texture_colour/convert/

# /*
#    Calculate RGB from HSV, reverse of RGB2HSV()
#    Hue is in degrees
#    Lightness is between 0 and 1
#    Saturation is between 0 and 1
# */
def hsv2rgb(hsv, alpha=None):
    # hsv = np.atleast_1d(hsv)
    # if (hsv.size == 1):
    #     hsv = angle2hsv(hsv)

    rgb = [0, 0, 0] # Could be a dict {}
    sat = [0, 0, 0]

    hsv[0] = hsv[0] % 360

    if (hsv[0] < 120):
        sat[0] = (120 - hsv[0]) / 60.0
        sat[1] = hsv[0] / 60.0
        sat[2] = 0
    elif (hsv[0] < 240):
        sat[0] = 0
        sat[1] = (240 - hsv[0]) / 60.0
        sat[2] = (hsv[0] - 120) / 60.0
    else:
        sat[0] = (hsv[0] - 240) / 60.0
        sat[1] = 0
        sat[2] = (360 - hsv[0]) / 60.0

    sat[0] = min(sat[0], 1)
    sat[1] = min(sat[1], 1)
    sat[2] = min(sat[2], 1)

    rgb[0] = int(round( (1 - hsv[1] + hsv[1] * sat[0]) * hsv[2] ))
    rgb[1] = int(round( (1 - hsv[1] + hsv[1] * sat[1]) * hsv[2] ))
    rgb[2] = int(round( (1 - hsv[1] + hsv[1] * sat[2]) * hsv[2] ))

    #alpha
    if (len(hsv) == 4):
        rgb.append(hsv[3])
    elif alpha:
        rgb.append(alpha)

    return rgb

def hsv_to_rgb(hsv, alpha=None):
    hsv = np.array(np.atleast_1d(hsv), dtype=np.uint8)
    rgb = np.array([0, 0, 0], dtype=np.uint8)
    sat = np.array([0, 0, 0], dtype=np.float16)

    hsv[0] = hsv[0] % 360

    sp = hsv[0] // 120   # 0..360 -> 0, 1 or 2

    sat[0] = ((sp + 1) * 120 - hsv[0])
    sat[1] = (hsv[0]         - sp * 120)
    #sat[2] = 0.0

    sat /= 60.0
    sat = np.fmin(sat, 1)

    # Move to the right part of spectrum (part = 0,1,2)
    sat = np.roll(sat, sp, 0)

    rgb = (1 - hsv[1] + hsv[1] * sat) * hsv[2]

    #alpha
    if (len(hsv) == 4):
        np.append(rgb, hsv[3])
    elif alpha:
        np.append(rgb, alpha)
    return rgb

# /*
#    Calculate HSV from RGB
#    Hue is in degrees
#    Lightness is betweeen 0 and 1
#    Saturation is between 0 and 1
# */
def rgb2hsv(rgb):
    hsv = [0, 0, 0] # Could be a dict {}

    themin = np.min(rgb)
    themax = np.max(rgb)
    delta = float(themax - themin)
    hsv[2] = themax # value
    hsv[1] = 0      # saturation
    if (themax > 0):
        hsv[1] = delta / themax

    hsv[0] = 0      # hue
    if (delta > 0):
        if (themax == rgb[0] and themax != rgb[1]):
            hsv[0] += (rgb[1] - rgb[0]) / delta
        if (themax == rgb[1] and themax != rgb[2]):
            hsv[0] += (2.0 + (rgb[2] - rgb[0]) / delta)
        if (themax == rgb[2] and themax != rgb[0]):
            hsv[0] += (4.0 + (rgb[0] - rgb[1]) / delta)
        hsv[0] *= 60.0

    hsv = map(lambda a: int(round(a)), hsv)

    #alpha
    if (len(rgb) == 4):
        hsv.append(rgb[3])

    return hsv

def angle2hsv(deg):
    # dtype uint8 loses precision, but doesn't matter here. It gets over a problem with hsv_to_rgb.
    return np.append(np.atleast_1d(deg % 360), np.array([1, 255, 255], dtype=np.uint8))

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

import types
types.signed = (int, float, np.signedinteger, np.floating)

def assert_type(types, *args):
    assert np.all(map(lambda p: isinstance(p, types), args)), \
        "All arguments must be instances of %s, got:\n%s" % (types, map(type, args))

def line_bresenham(x0, y0, x1, y1, colour=1.0, indices=False):
    """
    Bresenham line drawing algorithm.
    Converted from C version at http://free.pages.at/easyfilter/bresenham.html by Peter H.
    """
    assert_type(types.signed, x0, y0, x1, y1)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    dx =  np.abs(x1 - x0)
    dy = -np.abs(y1 - y0)
    bx = np.min((x0, x1))
    by = np.min((y0, y1))
    err = dx + dy # error value e_xy
    if indices:
        out = []
    else:
        out = np.zeros((-dy + 1, dx + 1))

    while True:
        if indices:
            out.append((x0, y0))
        else:
            out[y0 - by, x0 - bx] = colour
        if (x0 == x1 and y0 == y1):
            return np.array(out).T
        e2 = 2 * err
        if (e2 >= dy): err += dy; x0 += sx # e_xy + e_x > 0
        if (e2 <= dx): err += dx; y0 += sy # e_xy + e_y < 0

def line_linspace(x0, y0, x1, y1, endpoint=True):
    assert_type(types.signed, x0, y0, x1, y1)
    size = np.max([np.abs(x1 - x0), np.abs(y1 - y0)]) + int(bool(endpoint))
    return complex_as_reals(np.linspace(x0 + y0 * 1j, x1 + y1 * 1j, size, endpoint=endpoint)).astype(np.int32)

def line_linspace_cx(start, end, endpoint=True):
    assert_type(complex, start, end)
    distance = np.abs(start - end)
    size = np.max([distance.real, distance.imag]) + int(bool(endpoint))
    return np.round(complex_as_reals(np.linspace(start, end, size, endpoint=endpoint))).astype(np.int32)

def angles2hues(cx_samples, padding=False, loglevel=logging.ANIMA):
    """Converts angles of complex samples into hues"""

    # Get angles from points
    angles = np.angle(np.atleast_1d(cx_samples))
    logger.log(loglevel, "Angles:\n%s", repr(angles[:100]))

    # Get distances
    angles = pad(distances(angles), 0) if padding else distances(angles)

    # Get tau angles from points
    angles = (-np.abs( angles - (np.pi) ) % np.pi) / (2.0*np.pi)
    logger.log(loglevel, "Tau angles:\n%s", repr(angles[:100]))

    angles *= Sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Frequencies:\n%s", repr(angles[:100]))

    # Convert rad to deg
    low = np.log2(lowest_audible_hz)
    angles = ((np.log2(np.abs(angles)+1) - low) / 10 * 240) % 360  # 10 octaves mapped to red..violet
    logger.log(loglevel, "Scaled:\n%s\n", repr(angles[:100]))
    return angles

def chord_to_angle(length):
    """Return angle for chord length. Restrict to unit circle, ie. max length is 2.0"""
    # Limit highest freqs to Nyquist (blue or violet)
    d = np.fmin(np.abs(length), 2.0)
    # Limit lower freqs to lowest_audible_hz (red)
    d = np.fmax(d, 4.0*lowest_audible_hz/float(Sampler.rate))
    return np.arcsin(d / 2.0) * 2

def chord_to_hue(length):
    return deg(chord_to_angle(length))

def chord_to_tau(length):
    return chord_to_angle(length) / (2.0 * np.pi)

def tau_to_hue(tau, loglevel=logging.ANIMA):
    # Hue 240 is violet, and 8.96 is a factor for scaling back to 1.0
    #return (np.log2(np.abs(chord_to_tau(tau))+1)) * 8.96 * 240
    low = np.log2(lowest_audible_hz)
    taus = ((np.log2(np.abs(tau)+1) - low) / 8.96 * 240) % 360  # 10 octaves mapped to red..violet
    #logger.log(loglevel, "Scaled:\n%s\n", repr(taus[:100]))
    return taus

def chords_to_hues(signal, padding=True, loglevel=logging.ANIMA):
    phases = signal / np.fmax(np.abs(signal), minfloat(0.5)[0])

    # Get distances
    d = pad(distances(phases), -1) if padding else distances(phases)

    logger.log(loglevel, "%s Distances: %s", __name__, d)

    taus = np.apply_along_axis(chord_to_tau, 0, d) #np.append(d, d[-1])) # Append is a hack to get the same length back
    logger.log(loglevel, "%s Taus: %s", __name__, taus)

    #taus = taus / (2*np.pi) * Sampler.rate
    taus *= Sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Frequency median: %s", np.median(taus))
    logger.log(loglevel, "Frequencies:\n%s", repr(taus[:100]))
    #return taus

    hues = tau_to_hue(taus)
    return hues

@memoized
def get_huemap(steps = 6 * 255):
    #huemap = [ hsv2rgb(angle2hsv(angle)) for angle in np.arange(360) ] # step-size will become 4.2666... = 6 * 256 / 360.0
    huemap = [ tuple(hsv2rgb(angle2hsv(angle))) for angle in np.arange(0, 360, 1.0 / (steps / 360.0)) ] # len = 1536 = 360 * 4.2666...
    return np.array(huemap, dtype=np.uint8)

def colorize(samples, steps = 6 * 255, use_chords=True):
    #method = chords_to_hues if use_chords else angles2hues
    return get_huemap(steps)[(chords_to_hues(samples) * (steps / 360.0)).astype(np.int)]

def draw(samples, size=1000, dur=None, antialias=False, lines=False, axis=True, img=None, screen=None):
    """Draw the complex sound signal into specified size image."""
    # See http://jehiah.cz/archive/creating-images-with-numpy

    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(Sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner

    # Draw into existing img?
    if (img != None):
        size = img.shape[0]-1
    else:
        img = get_canvas(size, axis=axis)

    if dur:
        samples = samples[:int(round(dur * Sampler.rate))]

    # Clip -- amax could be just 'np.abs(np.max(samples))' for unit circle, but rectangular abs can be sqrt(2) > 1.0!
    amax = max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))
    if amax > 1.0:
        logger.warn("Clipping samples on draw() -- maximum magnitude was: %0.6f" % amax)
        samples = clip(samples)

    if lines:
        if antialias: # Colorize
            # raise exceptions.NotImplementedError("Drawing lines with antialias not implemented yet.")
            # TODO: optimize colours with lines!
            #scaled = (samples + 1+1j) / (2.0 * size)

            if True: #lines and not antialias: #(img != None):
                colors = colorize(samples)
                pts = get_points(samples, size).T
                for (i, ends) in enumerate(pairwise(pts)):
                    #pts = get_points(np.array(ends), size).transpose()
                    #color = hsv2rgb(angle2hsv(chords_to_hues(ends, padding=False)))
                    #color = pygame.Color(*list(hsv2rgb(angle2hsv(chords_to_hues(ends, padding=False))))[:-1])
                    pygame.draw.aaline(screen, colors[i], *ends)
            else:
                colors = colorize(samples)
                pts = pad(samples, -1)
                for i in xrange(len(samples)):
                    line = line_linspace_cx(pts[i], pts[i + 1], endpoint=False)
                    img[line[0], line[1]] = colors[i][-1] # Drop alpha

        else:
            # raise exceptions.NotImplementedError("Drawing lines without antialias not implemented yet.")
            pts = get_points(samples, size).transpose()
            pygame.draw.aalines(screen, pygame.Color('orange'), False, pts, 1)
    else:
        points = get_points(samples, size)
        if antialias:
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

            colors = colorize(samples) # or 255 for greyscale

            img[pos[0][1], pos[0][0], :] += colors * np.repeat(values_11, 4).reshape(len(samples), 4)
            img[pos[1][1], pos[1][0], :] += colors * np.repeat(values_10, 4).reshape(len(samples), 4)
            img[pos[2][1], pos[2][0], :] += colors * np.repeat(values_01, 4).reshape(len(samples), 4)
            img[pos[3][1], pos[3][0], :] += colors * np.repeat(values_00, 4).reshape(len(samples), 4)
        else:
            points = np.cast['uint32'](points)  # 0 to 599
            img[points[0], (size - 1) - points[1]] = colorize(samples)
        
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

def show(img, plot=False):
    if (plot and plt):
        imgplot = plt.imshow(img[:,:,0])
        imgplot.set_cmap('hot')
    else:
        image = Image.fromarray(img, 'RGBA')
        image.show()
    return False

def fast_graph(samples, size=1000, plot=False):
    return graph(samples, size, plot, antialias=False)

def graph(samples, size=1000, dur=None, plot=False, axis=True, antialias=False, lines=False):
    if dur:
        samples = samples[:int(round(dur * Sampler.rate))]
    img = draw(samples, size=size, antialias=antialias, lines=lines, axis=axis).transpose((1, 0, 2))
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

