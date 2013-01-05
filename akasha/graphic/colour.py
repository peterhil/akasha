#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colour conversions and other colour utility functions.
"""


import logging
import numpy as np

from akasha.timing import sampler
from akasha.types import colour_values, colour_result
from akasha.utils.decorators import memoized
from akasha.utils.log import logger
from akasha.utils.math import distances, minfloat, pad, rad_to_deg


lowest_audible_hz = 16.35
white = np.array([255, 255, 255, 255])


# Colour conversion functions hsv2rgb and rgb2hsv ported to Python from C sources at:
# http://paulbourke.net/texture_colour/colourspace/
# Section: "HSV Colour space" / "C code to transform between RGB and HSV is given below"


def hsv2rgb(hsv, alpha=None, dtype=colour_result):
    """
    Calculate RGB from HSV.

    Hue is in degrees between 0 and 360
    Lightness is between 0 and 1
    Saturation is between 0 and 1
    """
    hsv = np.array(np.atleast_1d(hsv), dtype=colour_values)

    rgb = np.array([0, 0, 0], dtype=colour_values)
    sat = np.array([0, 0, 0], dtype=colour_values)

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

    rgb[0] = (1 - hsv[1] + hsv[1] * sat[0]) * hsv[2]
    rgb[1] = (1 - hsv[1] + hsv[1] * sat[1]) * hsv[2]
    rgb[2] = (1 - hsv[1] + hsv[1] * sat[2]) * hsv[2]

    # Alpha
    if (len(hsv) == 4):
        np.append(rgb, hsv[3])
    elif alpha:
        np.append(rgb, alpha)

    return rgb.astype(dtype)


def hsv_to_rgb(hsv, alpha=None, dtype=colour_result):
    """
    Calculate RGB from HSV using Numpy.

    Hue is in degrees between 0 and 360
    Lightness is between 0 and 1
    Saturation is between 0 and 1
    """
    hsv = np.array(np.atleast_1d(hsv), dtype=colour_values)
    rgb = np.array([0, 0, 0], dtype=colour_values)
    sat = np.array([0, 0, 0], dtype=colour_values)

    hsv[0] = hsv[0] % 360

    sp = hsv[0] // 120   # 0..360 -> 0, 1 or 2

    sat[0] = ((sp + 1) * 120 - hsv[0])
    sat[1] = (hsv[0] - sp * 120)
    #sat[2] = 0.0

    sat /= 60.0
    sat = np.fmin(sat, 1)

    # Move to the right part of spectrum (part = 0,1,2)
    sat = np.roll(sat, sp, 0)

    rgb = (1 - hsv[1] + hsv[1] * sat) * hsv[2]

    # Alpha
    if (len(hsv) == 4):
        np.append(rgb, hsv[3])
    elif alpha:
        np.append(rgb, alpha)

    return rgb.astype(dtype)


def rgb2hsv(rgb, dtype=colour_result):
    """
    Calculate HSV from RGB.

    Hue is in degrees
    Lightness is betweeen 0 and 1
    Saturation is between 0 and 1
    """
    rgb = np.array(np.atleast_1d(rgb), dtype=colour_values)
    hsv = np.array([0, 0, 0], dtype=colour_values)

    themin = np.min(rgb)
    themax = np.max(rgb)
    delta = float(themax - themin)
    hsv[2] = themax  # value
    hsv[1] = 0       # saturation
    if (themax > 0):
        hsv[1] = delta / themax

    hsv[0] = 0       # hue
    if (delta > 0):
        if (themax == rgb[0] and themax != rgb[1]):
            hsv[0] += (rgb[1] - rgb[0]) / delta
        if (themax == rgb[1] and themax != rgb[2]):
            hsv[0] += (2.0 + (rgb[2] - rgb[0]) / delta)
        if (themax == rgb[2] and themax != rgb[0]):
            hsv[0] += (4.0 + (rgb[0] - rgb[1]) / delta)
        hsv[0] *= 60.0

    # Alpha
    if (len(rgb) == 4):
        hsv.append(rgb[3])

    return hsv.astype(dtype)


def angle2hsv(angles, dtype=colour_result):
    """
    Convert hue angle to rgb colour.
    """
    # dtype uint8 loses precision, but doesn't matter here.
    # It gets over a problem with hsv_to_rgb.
    return np.append(
        np.atleast_1d(angles % 360),
        np.array([1, 255, 255], dtype=colour_values)
    ).astype(dtype)


def angles2hues(cx_samples, padding=True, loglevel=logging.ANIMA):
    """
    Convert angles of complex samples into hues.
    """
    # Get angles from points
    angles = np.angle(np.atleast_1d(cx_samples))
    logger.log(loglevel, "Angles:\n%s", repr(angles[:100]))

    # Get distances
    angles = pad(distances(angles), 0) if padding else distances(angles)

    # Get tau angles from points
    angles = (-np.abs(angles - (np.pi)) % np.pi) / (2.0 * np.pi)
    logger.log(loglevel, "Tau angles:\n%s", repr(angles[:100]))

    angles *= sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Frequencies:\n%s", repr(angles[:100]))

    # Convert rad to deg
    low = np.log2(lowest_audible_hz)

    # 10 octaves mapped to red..violet
    angles = ((np.log2(np.abs(angles) + 1) - low) / 8.96 * 240) % 360

    logger.log(loglevel, "Scaled:\n%s\n", repr(angles[:100]))
    return angles


def chord_to_angle(length):
    """
    Return radian angle of a point on unit circle with the specified chord length from 1+0j.
    Restrict to unit circle, ie. max length is 2.0.
    """
    # Limit highest freqs to Nyquist (blue or violet)
    d = np.fmin(np.abs(length), 2.0)
    # Limit lower freqs to lowest_audible_hz (red)
    d = np.fmax(d, 4.0 * lowest_audible_hz / float(sampler.rate))
    return np.arcsin(d / 2.0) * 2


def chord_to_hue(length):
    """
    Return degrees from a chord length between a point on unit circle and 1+0j.
    """
    return rad_to_deg(chord_to_angle(length))


def chord_to_tau(length):
    """
    Return tau angle from a chord length between a point on unit circle and 1+0j.
    """
    return chord_to_angle(length) / (2.0 * np.pi)


def tau_to_hue(tau_angles):
    """
    Return hue angles (in degrees) from tau angles.
    """
    # Hue 240 is violet, and 8.96 is a factor for scaling back to 1.0
    #return (np.log2(np.abs(chord_to_tau(tau_angles))+1)) * 8.96 * 240
    low = np.log2(lowest_audible_hz)

    # 10 octaves mapped to red..violet
    return ((np.log2(np.abs(tau_angles) + 1) - low) / 8.96 * 240) % 360


def chords_to_hues(signal, padding=True, loglevel=logging.ANIMA):
    """
    Return hue angles from instantaneous frequencies of a signal.
    """
    phases = signal / np.fmax(np.abs(signal), minfloat(0.5)[0])

    # Get distances
    d = pad(distances(phases), -1) if padding else distances(phases)

    logger.log(loglevel, "%s Distances: %s", __name__, d)

    # Append is a hack to get the same length back
    taus = np.apply_along_axis(chord_to_tau, 0, d)  # np.append(d, d[-1]))
    logger.log(loglevel, "%s Taus: %s", __name__, taus)

    #taus = taus / (2*np.pi) * sampler.rate
    taus *= sampler.rate  # 0..Fs/2
    logger.log(loglevel, "Frequency median: %s", np.median(taus))
    logger.log(loglevel, "Frequencies:\n%s", repr(taus[:100]))

    return tau_to_hue(taus)


@memoized
def get_huemap(steps=6 * 255):
    """
    Get a hue map (a spectrum) of n evenly spaced steps.
    Returns an array of rgb colours.
    """
    # step-size will become 4.2666... = 6 * 256 / 360.0
    # huemap = [hsv2rgb(angle2hsv(angle)) for angle in np.arange(360)]

    # len = 1536 = 360 * 4.2666...
    angles = np.arange(0, 360, 1.0 / (steps / 360.0))
    huemap = [tuple(hsv2rgb(angle2hsv(angle))) for angle in angles]

    return np.array(huemap, dtype=np.uint8)


def colorize(signal, steps=6 * 255, use_chords=True):
    """
    Colorize a signal according to it's instantaneous frequency.
    """
    colourizer = chords_to_hues if use_chords else angles2hues
    return get_huemap(steps)[(colourizer(signal) * (steps / 360.0)).astype(np.int)]
