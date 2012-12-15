#!/usr/bin/env python
# -*- coding: utf-8 -*-

import colorsys
import numpy as np


colour_values = np.float32
colour_result = np.float32


# Colour conversions from: http://local.wasp.uwa.edu.au/~pbourke/texture_colour/convert/

# /*
#    Calculate RGB from HSV, reverse of RGB2HSV()
#    Hue is in degrees
#    Lightness is between 0 and 1
#    Saturation is between 0 and 1
# */
def hsv2rgb(hsv, alpha=None, dtype=colour_result):
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
    hsv = np.array(np.atleast_1d(hsv), dtype=colour_values)
    rgb = np.array([0, 0, 0], dtype=colour_values)
    sat = np.array([0, 0, 0], dtype=colour_values)

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

    # Alpha
    if (len(hsv) == 4):
        np.append(rgb, hsv[3])
    elif alpha:
        np.append(rgb, alpha)

    return rgb.astype(dtype)

# /*
#    Calculate HSV from RGB
#    Hue is in degrees
#    Lightness is betweeen 0 and 1
#    Saturation is between 0 and 1
# */
def rgb2hsv(rgb, dtype=colour_result):
    rgb = np.array(np.atleast_1d(rgb), dtype=colour_values)
    hsv = np.array([0, 0, 0], dtype=colour_values)

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

    # Alpha
    if (len(rgb) == 4):
        hsv.append(rgb[3])

    return hsv.astype(dtype)

def angle2hsv(deg, dtype=colour_result):
    # dtype uint8 loses precision, but doesn't matter here. It gets over a problem with hsv_to_rgb.
    return np.append(np.atleast_1d(deg % 360), np.array([1, 255, 255], dtype=colour_values)).astype(dtype)

