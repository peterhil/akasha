#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from cmath import phase
from timing import Sampler
from PIL import Image

from utils.math import normalize, clip

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
    return [deg % 360, 1, 255, 255]

def hist_graph(samples, size=1000):
    """Uses numpy histogram2d to make an image from complex signal."""
    c = samples.view(np.float64).reshape(len(samples), 2).transpose()
    x, y = c[0], -c[1]
    hist, y_edges, x_edges = np.histogram2d(y, x, bins=size, range=[[-1.,1.],[-1.,1.]], normed=False)
    image = Image.fromarray(np.array(hist / hist.mean() * 255, dtype=np.uint8),'L')
    image.show()

def get_canvas(size=1000):
    return np.zeros((size+1,size+1,4), np.uint8)             # Note: y, x

def get_points(samples, size=1000):
    # Scale to size and interpret values as pixel centers
    samples = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))      # 0.5 to 599.5

    # Convert complex samples to real number coordinate points
    return samples.view(np.float).reshape(len(samples), 2).transpose()    # 0.5 to 599.5

def draw(samples, size=1000, antialias=True):
    """Draw the complex sound signal into specified size image."""
    # See http://jehiah.cz/archive/creating-images-with-numpy

    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(Sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner

    img = get_canvas(size)
    points = get_points(samples, size)

    ## Angles for hues

    # Get angles from points
    angles = np.array(map(phase, samples)) + np.pi
    # print repr(angles[:100])

    # Get diffs & convert to degrees 0..240 (red..blue)
    angles = (np.abs(np.append(angles[-1], angles[:-1]) - angles) % (2.0 * np.pi))  # Fixme! First samples should not be handled differently!
    # print repr(angles[:100])

    angles =  angles / (2.0 * np.pi) * Sampler.rate  # 0..Fs/2
    # print repr(angles[:100])

    # Convert rad to deg
    low = np.log2(lowest_audible_hz)
    angles = ((np.log2(np.abs(angles)+1) - low) * 24) % 360   # (np.log2(f)-low) / 10 * 240  red..violet
    # angles = angles * 240.0     # red..violet
    # print repr(angles[:100])

    # Draw axis
    img[size/2.0,:] = img[:,size/2.0] = [42,42,42,127]

    if antialias:
        # Draw with antialising
        centers = np.round(points)  # 1.0 to 600.0
        bases = np.cast['uint64'](centers) - 1   # 0 to 599
        deltas = points - bases - 0.5

        values_00 = deltas[1] * deltas[0]
        values_01 = deltas[1] * (1.0 - deltas[0])
        values_10 = (1.0 - deltas[1]) * deltas[0]
        values_11 = (1.0 - deltas[1]) * (1.0 - deltas[0])

        img[(size-1) - bases[1], bases[0], :] += np.repeat((values_11 * 255), 4).reshape(len(samples),4)
        img[(size-1) - bases[1], bases[0]+1, :] += np.repeat((values_10 * 255), 4).reshape(len(samples),4)
        img[(size-1) - (bases[1]+1), bases[0], :] += np.repeat((values_01 * 255), 4).reshape(len(samples),4)
        img[(size-1) - (bases[1]+1), bases[0]+1, :] += np.repeat((values_00 * 255), 4).reshape(len(samples),4)
    else:
        # Cast floats to integers
        points = np.cast['uint64'](points)  # 0 to 599

        # Draw image
        img[(size - 1) - points[1], points[0]] = map(lambda c: hsv2rgb(angle2hsv(c)), angles)     #[255,255,255,255]

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

def graph(samples, size=1000, plot=False, antialias=True):
    img = draw(samples, size, antialias)
    show(img, plot)
    return False

def plot(samples):
    "Plot samples using matplotlib"
    imgplot = plt.imshow([samples.real, samples.imag])
    imgplot.set_cmap('hot')
    return False
