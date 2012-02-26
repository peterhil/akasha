#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from cmath import phase
import colorsys
from timing import Sampler
from PIL import Image
from exceptions import ValueError
import logging

from utils.decorators import memoized
from utils.math import normalize, clip, deg, distances, pad
from utils.log import logger

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except:
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

def get_canvas(size=1000, axis=True):
    img = np.zeros((size+1,size+1,4), np.uint8)             # Note: y, x
    if axis:
        # Draw axis
        img[size/2.0,:] = img[:,size/2.0] = [42,42,42,127]
    return img

def get_points(samples, size=1000):
    # Scale to size and interpret values as pixel centers
    samples = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))      # 0.5 to 599.5

    # Convert complex samples to real number coordinate points
    return samples.view(np.float).reshape(len(samples), 2).transpose()    # 0.5 to 599.5

def angles2hues(cx_samples, padding=True, loglevel=logging.ANIMA):
    """Converts angles of complex samples into hues"""
    cx_samples = np.atleast_1d(cx_samples)

    # Get angles from points
    angles = np.angle(cx_samples)
    logger.log(loglevel, "Angles:\n%s", repr(angles[:100]))

    # Get distances
    angles = pad(distances(angles), 0) if padding else distances(angles)

    # Get tau angles from points
    angles = (-np.abs( angles - (np.pi) ) % np.pi) / (2.0*np.pi)
    logger.log(loglevel, "Tau angles:\n%s", repr(angles[:100]))

    angles *= Sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Tau angles * Fs -> 0..Fs:\n%s", repr(angles[:100]))

    # Convert rad to deg
    low = np.log2(lowest_audible_hz)
    angles = ((np.log2(np.abs(angles)+1) - low) * 24) % 360   # (np.log2(f)-low) / 10 * 240  red..violet
    # angles = angles * 240.0     # red..violet
    logger.log(loglevel, "Scaled:\n%s\n", repr(angles[:100]))
    return angles

def phase2hues(cx_samples, padding=True, loglevel=logging.ANIMA):
    """Converts angles of complex samples into hues"""
    cx_samples = np.atleast_1d(cx_samples)
    
    # Get angles from points
    angles = np.angle(cx_samples) + np.pi
    logger.log(loglevel, "Angle + pi:\n%s", repr(angles[:100]))

    # Get distances
    angles = pad(distances(angles), 0) if padding else distances(angles)

    # Convert to degrees 0..240 (red..blue)
    angles = angles % (2.0 * np.pi)
    logger.log(loglevel, "Diffs modulo 2pi:\n%s", repr(angles[:100]))

    angles = angles / (2.0 * np.pi) * Sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Rad to tau * Fs -> 0..Fs:\n%s", repr(angles[:100]))

    # Convert rad to deg
    low = np.log2(lowest_audible_hz)
    angles = ((np.log2(np.abs(angles)+1) - low) * 24) % 360   # (np.log2(f)-low) / 10 * 240  red..violet
    # angles = angles * 240.0     # red..violet
    logger.log(loglevel, "Scaled:\n%s\n", repr(angles[:100]))
    return angles

def chord_to_angle(length):
    """Return angle for chord length. Restrict to unit circle, ie. max length is 2.0"""
    return np.arcsin(np.fmin(np.abs(length), 2.0) / 2.0) * 2

def chord_to_hue(length):
    return deg(chord_to_angle(length))

def chord_to_tau(length):
    return chord_to_angle(length) / (2.0 * np.pi)

def tau_to_hue(tau):
    # Hue 240 is violet, and 8.96 is a factor for scaling back to 1.0
    return (np.log2(np.abs(chord_to_tau(tau))+1)) * 8.96 * 240

def chords_to_hues(signal):
    d = distances(signal)
    logger.debug("%s Distances: %s", __name__, d)
    taus = np.apply_along_axis(chord_to_tau, 0, np.append(d, d[-1])) # Append is a hack to get the same length back
    hues = tau_to_hue(taus)
    logger.debug("%s Hues: %s", __name__, hues)
    return hues

@memoized
def getHuemap(steps = 6 * 255):
    #huemap = [ hsv2rgb(angle2hsv(angle)) for angle in np.arange(360) ] # step-size will become 4.2666... = 6 * 256 / 360.0
    huemap = [ tuple(hsv2rgb(angle2hsv(angle))) for angle in np.arange(0, 360, 1.0 / (steps / 360.0)) ] # len = 1536 = 360 * 4.2666...
    return np.array(huemap, dtype=np.uint8)

def colorize(samples, steps = 6 * 255):
    return getHuemap(steps)[(phase2hues(samples) * (steps / 360.0)).astype(np.int)]

def draw(samples, size=1000, antialias=False, axis=True, img=None):
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
        img = get_canvas(size, axis)

    # Clip -- amax could be just 'np.abs(np.max(samples))' for unit circle, but rectangular abs can be sqrt(2) > 1.0!
    amax = max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))
    if amax > 1.0:
        logger.warn("Clipping samples on draw() -- maximum magnitude was: %0.6f" % amax)
        samples = clip(samples)

    points = get_points(samples, size)

    if antialias:
        # Draw with antialising
        centers = np.round(points)  # 1.0 to 600.0
        bases = np.cast['uint32'](centers) - 1   # 0 to 599
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

        img[pos[0][0], pos[0][1], :] += colors * np.repeat(values_11, 4).reshape(len(samples), 4)
        img[pos[1][0], pos[1][1], :] += colors * np.repeat(values_10, 4).reshape(len(samples), 4)
        img[pos[2][0], pos[2][1], :] += colors * np.repeat(values_01, 4).reshape(len(samples), 4)
        img[pos[3][0], pos[3][1], :] += colors * np.repeat(values_00, 4).reshape(len(samples), 4)
    else:
        # Cast floats to integers
        points = np.cast['uint32'](points)  # 0 to 599

        # Draw image
        
        # img[(size - 1) - points[1], points[0]] = np.apply_along_axis(hsv2rgb, 0, phase2hues(samples))
        # img[(size - 1) - points[1], points[0]] = map(lambda c: hsv2rgb(angle2hsv(c)), phase2hues(samples))
        # img[(size - 1) - points[1], points[0]] = map(lambda c: hsv2rgb(angle2hsv(c)), hues)
        # img[(size - 1) - points[1], points[0]] = map(lambda c: np.append(colorsys.hsv_to_rgb(*np.append([c/360.0], [1, 255])), [255]), hues)

        # Draw image v2
        
        # hues = angles2hues(samples)
        # hsv_colours = np.repeat(np.array([[0,1,255]], dtype=np.float32), hues.size, 0)
        # hsv_colours[:,0] = hues / 360.0
        # 
        # rgb_colours = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), 1, hsv_colours)
        # rgb_colours = np.insert(rgb_colours.T, [1], 255.0, 0).T # Append alpha
        # 
        # img[(size - 1) - points[1], points[0]] = rgb_colours

        # Draw image v3
        img[(size - 1) - points[1], points[0]] = colorize(samples)
        
    return img


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

def graph(samples, size=1000, plot=False, axis=True, antialias=False):
    img = draw(samples, size, antialias, axis)
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